"""
LLM Runtime for Structured Calls with Validation and Retries

This module provides a robust runtime for making structured LLM calls
with automatic validation, retries, and error handling.
"""

import json
import time
import uuid
import logging
from typing import Type, TypeVar, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import re

from .schemas import VetoResolutionSummary, LLMProcessingResult

# Set up logging
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class StructuredLLMClient:
    """
    Client for making structured LLM calls with validation and retries.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_retries: int = 3):
        """
        Initialize the structured LLM client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for completions
            max_retries: Maximum number of retries on validation failure
        """
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        self.model = model
        self.max_retries = max_retries
    
    def structured_llm_call(
        self,
        prompt: str,
        schema_class: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        correlation_id: Optional[str] = None
    ) -> LLMProcessingResult:
        """
        Make a structured LLM call with validation and retries.
        
        Args:
            prompt: User prompt
            schema_class: Pydantic model class for validation
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            LLMProcessingResult with success/failure information
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        start_time = time.time()
        retry_count = 0
        
        # Generate JSON schema for the model
        json_schema = schema_class.model_json_schema()
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(schema_class.__name__)
        
        # Enhanced system prompt with schema
        enhanced_system_prompt = f"""{system_prompt}

CRITICAL: You must respond with ONLY valid JSON that matches this exact schema.

Required JSON structure:
{{
  "resolution_title": "string (10-500 chars)",
  "description": "string (20-150 chars max - VERY IMPORTANT)",
  "applicant_country": "string or null (for membership applications only)",
  "historical_context": "string or null (max 200 chars)",
  "resolution_type": "one of: membership_application, territorial_dispute, peacekeeping_operation, sanctions_regime, humanitarian_intervention, arms_control, human_rights, other",
  "confidence_score": integer (1-100),
  "contains_country_name": boolean,
  "is_descriptive_only": boolean
}}

CRITICAL RULES:
1. Respond with JSON ONLY - no markdown, no explanations, no additional text
2. ALL fields above must be present (use null for optional fields if unknown)
3. "description" must be 20-150 characters maximum - this is strictly enforced
4. "resolution_type" must be exactly one of the listed values
5. "confidence_score" must be an integer between 1-100
6. Do NOT mention vetoes, opposition, or blocking in the description
7. Focus on what the resolution was about, not who opposed it

Full schema for reference:
{json.dumps(json_schema, indent=2)}"""
        
        while retry_count <= self.max_retries:
            try:
                logger.info(f"Making LLM call (attempt {retry_count + 1}/{self.max_retries + 1}) - {correlation_id}")
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )
                
                # Extract and clean the response
                raw_content = response.choices[0].message.content.strip()
                cleaned_content = self._clean_json_response(raw_content)
                
                logger.debug(f"Raw LLM response: {raw_content[:200]}...")
                logger.debug(f"Cleaned JSON: {cleaned_content[:200]}...")
                
                # Parse JSON
                try:
                    json_data = json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response: {e}")
                
                # Validate with Pydantic
                validated_result = schema_class(**json_data)
                
                # Success!
                processing_time = int((time.time() - start_time) * 1000)
                
                return LLMProcessingResult(
                    success=True,
                    result=validated_result,
                    retry_count=retry_count,
                    processing_time_ms=processing_time,
                    correlation_id=correlation_id,
                    model_used=self.model
                )
                
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                retry_count += 1
                error_msg = str(e)
                
                logger.warning(f"Validation error (attempt {retry_count}/{self.max_retries + 1}) - {correlation_id}: {error_msg}")
                
                if retry_count > self.max_retries:
                    # Max retries exceeded
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    return LLMProcessingResult(
                        success=False,
                        error_message=f"Max retries exceeded. Last error: {error_msg}",
                        retry_count=retry_count - 1,
                        processing_time_ms=processing_time,
                        correlation_id=correlation_id,
                        model_used=self.model
                    )
                
                # Prepare retry with error feedback
                enhanced_system_prompt += f"""

PREVIOUS ATTEMPT FAILED with error: {error_msg}

Please fix the issue and provide valid JSON that strictly follows the schema."""
                
                # Brief pause before retry
                time.sleep(0.5)
                
            except Exception as e:
                # Unexpected error
                processing_time = int((time.time() - start_time) * 1000)
                logger.error(f"Unexpected error in LLM call - {correlation_id}: {e}")
                
                return LLMProcessingResult(
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                    retry_count=retry_count,
                    processing_time_ms=processing_time,
                    correlation_id=correlation_id,
                    model_used=self.model
                )
    
    def _clean_json_response(self, raw_response: str) -> str:
        """
        Clean the LLM response to extract valid JSON.
        
        Args:
            raw_response: Raw response from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', raw_response)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)
        
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Remove any explanatory text before or after JSON
        lines = cleaned.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('{'):
                in_json = True
                brace_count += stripped_line.count('{') - stripped_line.count('}')
                json_lines.append(line)
            elif in_json:
                brace_count += stripped_line.count('{') - stripped_line.count('}')
                json_lines.append(line)
                if brace_count <= 0:
                    break
        
        if json_lines:
            cleaned = '\n'.join(json_lines)
        else:
            # Fallback: try to find JSON object boundaries
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx + 1]
        
        return cleaned
    
    def _get_default_system_prompt(self, schema_name: str) -> str:
        """
        Get default system prompt based on schema type.
        
        Args:
            schema_name: Name of the Pydantic schema class
            
        Returns:
            Default system prompt
        """
        if schema_name == "VetoResolutionSummary":
            return """You are a UN Security Council expert specializing in analyzing vetoed resolutions.

Your task is to provide structured, factual summaries of UN Security Council resolutions that were vetoed.

CRITICAL GUIDELINES:
- Keep descriptions between 20-150 characters (strictly enforced)
- Be completely neutral and factual
- Focus ONLY on what the resolution addressed, not who opposed it
- For membership applications, extract the specific country name for "applicant_country" field
- Never mention "vetoed", "opposed", "blocked", "rejected" in descriptions
- Use present tense and active voice
- Be concise but informative

EXAMPLES:
- Good: "Resolution addressed Kuwait's application for UN membership."
- Bad: "Resolution to admit Kuwait was vetoed by the Soviet Union."
- Good: "Resolution concerned territorial dispute in Kashmir region."
- Bad: "Resolution opposed by China regarding Kashmir conflict."

For membership applications:
- Set resolution_type to "membership_application"
- Extract country name for applicant_country field
- Description should mention "application for UN membership"

Always provide all required fields with appropriate values."""
        
        return "You are an expert analyst providing structured information based on the given data."


def create_veto_summary_client(api_key: str) -> StructuredLLMClient:
    """
    Factory function to create a client specifically for veto summaries.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        Configured StructuredLLMClient
    """
    return StructuredLLMClient(
        api_key=api_key,
        model="gpt-4o-mini",
        max_retries=3
    )


def generate_veto_summary_structured(
    resolution_title: str,
    year: int,
    vetoing_countries: list,
    api_key: str,
    correlation_id: Optional[str] = None
) -> LLMProcessingResult:
    """
    Generate a structured veto summary using the LLM client with enhanced historical context.
    
    Args:
        resolution_title: Full resolution title
        year: Year of the veto
        vetoing_countries: List of countries that vetoed
        api_key: OpenAI API key
        correlation_id: Optional correlation ID
        
    Returns:
        LLMProcessingResult with VetoResolutionSummary
    """
    client = create_veto_summary_client(api_key)
    
    # Create enhanced prompt with historical context
    vetoing_countries_str = ", ".join(vetoing_countries)
    
    # Determine historical context based on year and title patterns
    historical_context = _get_historical_context_hint(resolution_title, year)
    
    prompt = f"""You are a UN Security Council historian with deep knowledge of Cold War politics, decolonization, and international conflicts. Analyze this vetoed resolution using your historical expertise:

RESOLUTION DETAILS:
Title: "{resolution_title}"
Year: {year}
Vetoed by: {vetoing_countries_str}

HISTORICAL CONTEXT HINT:
{historical_context}

ANALYSIS REQUIREMENTS:
1. RESOLUTION TITLE: Return the provided title as-is, but if it exceeds 1400 characters, truncate it intelligently at a natural break point and add "..." at the end.

2. DESCRIPTION (20-150 chars): Write a precise, informative description of what this resolution addressed. Use your knowledge of the historical period, conflicts, and political situations of {year}.

3. COUNTRY IDENTIFICATION: For membership applications, extract the specific country name. Look for patterns like "Admission of [Country]", "Application for Membership ([Country])", or country names in parentheses.

4. RESOLUTION TYPE: Categorize based on content:
   - membership_application: UN membership requests
   - territorial_dispute: Border conflicts, territorial claims
   - peacekeeping_operation: UN peacekeeping deployments
   - sanctions_regime: Economic or diplomatic sanctions
   - humanitarian_intervention: Humanitarian crises, refugee situations
   - arms_control: Nuclear/weapons proliferation, disarmament
   - human_rights: Apartheid, civil rights, political persecution
   - other: General political situations

5. HISTORICAL CONTEXT: Provide relevant background about why this issue was contentious in {year}.

6. CONFIDENCE: Rate your certainty (1-100) based on your historical knowledge and the clarity of the resolution title.

EXAMPLES OF GOOD DESCRIPTIONS:
- "Resolution addressed Bangladesh's application for UN membership during the 1971 independence war."
- "Resolution concerned Soviet intervention in Afghanistan following the 1979 invasion."
- "Resolution addressed South African apartheid policies and international sanctions."
- "Resolution concerned the Greek civil war and foreign intervention in 1946."
- "Resolution addressed Greek civil war situation following Ukrainian SSR telegram in 1946."

CRITICAL INSTRUCTIONS:
- Use your extensive historical knowledge about {year}
- Be specific about conflicts, countries, and political situations
- Never mention vetoes, opposition, or blocking in the description
- Focus on the substantive issue the resolution addressed
- If the title mentions a country in parentheses, that's usually the subject matter, not a geographic error
- SPECIAL CASE: Ukrainian SSR telegrams in 1946 were typically about Greek civil war affairs, not Ukraine itself
- TITLE HANDLING: If the resolution title is extremely long (>1400 chars), truncate it at a natural break and add "..."
- NEUTRAL LANGUAGE: Use factual, neutral language. "Complaint against" or "situation regarding" are acceptable neutral terms"""
    
    return client.structured_llm_call(
        prompt=prompt,
        schema_class=VetoResolutionSummary,
        correlation_id=correlation_id
    )


def _get_historical_context_hint(resolution_title: str, year: int) -> str:
    """
    Generate historical context hints based on resolution title and year.
    
    Args:
        resolution_title: Full resolution title
        year: Year of the veto
        
    Returns:
        Historical context hint string
    """
    title_lower = resolution_title.lower()
    
    # Cold War period contexts
    if 1946 <= year <= 1991:
        if "spanish" in title_lower:
            return f"In {year}, Spain was under Franco's dictatorship, and the UN was debating its international status post-WWII."
        elif "greek" in title_lower or "greece" in title_lower or ("ukrainian ssr" in title_lower and year == 1946):
            return f"In {year}, Greece was experiencing civil war between communist and government forces, with international implications. Ukrainian SSR telegrams often concerned Greek affairs."
        elif "ukraine" in title_lower and year >= 2014:
            return f"In {year}, Ukraine was dealing with territorial conflicts and Russian intervention following the 2014 crisis."
        elif "middle east" in title_lower or "palestinian" in title_lower:
            return f"In {year}, the Middle East was experiencing ongoing Arab-Israeli conflicts and Palestinian territorial disputes."
        elif "admission" in title_lower or "membership" in title_lower:
            return f"In {year}, UN membership was often contested due to Cold War politics and decolonization processes."
        elif "south africa" in title_lower:
            return f"In {year}, South Africa's apartheid system was a major international concern and subject of sanctions."
        elif "southern rhodesia" in title_lower or "rhodesia" in title_lower:
            return f"In {year}, Southern Rhodesia (now Zimbabwe) was dealing with independence struggles and minority rule issues."
        elif "bangladesh" in title_lower:
            return f"In {year}, Bangladesh was seeking independence from Pakistan, creating regional tensions."
        elif "angola" in title_lower:
            return f"In {year}, Angola was experiencing civil war and decolonization from Portuguese rule."
        elif "vietnam" in title_lower:
            return f"In {year}, Vietnam was divided and experiencing conflict, with Cold War implications."
        elif "korea" in title_lower or "dprk" in title_lower:
            return f"In {year}, Korea remained divided with ongoing tensions and nuclear concerns."
        elif "non-proliferation" in title_lower:
            return f"In {year}, nuclear proliferation was a major international security concern."
        elif "sudan" in title_lower:
            return f"In {year}, Sudan was experiencing internal conflicts and humanitarian crises."
        elif "mali" in title_lower:
            return f"In {year}, Mali was dealing with security challenges and international intervention needs."
    
    # Post-Cold War contexts
    elif year >= 1992:
        if "ukraine" in title_lower:
            return f"In {year}, Ukraine was dealing with territorial integrity issues and Russian aggression."
        elif "middle east" in title_lower or "palestinian" in title_lower:
            return f"In {year}, the Middle East continued to experience conflicts related to Palestinian statehood and regional tensions."
        elif "syria" in title_lower:
            return f"In {year}, Syria was experiencing civil war and humanitarian crisis."
        elif "iraq" in title_lower:
            return f"In {year}, Iraq was dealing with post-invasion reconstruction and security challenges."
        elif "afghanistan" in title_lower:
            return f"In {year}, Afghanistan was experiencing ongoing conflict and international intervention."
    
    # General context
    return f"In {year}, this resolution addressed international peace and security concerns typical of the era's geopolitical tensions."

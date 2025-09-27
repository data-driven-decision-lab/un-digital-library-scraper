"""
LLM Module for Structured Security Council Analysis

This module provides structured LLM interactions with Pydantic validation
for Security Council veto analysis and description generation.
"""

from .schemas import (
    VetoResolutionSummary,
    BatchVetoSummaries,
    LLMProcessingResult,
    EnhancedVetoRecord
)

from .runtime import (
    StructuredLLMClient,
    create_veto_summary_client,
    generate_veto_summary_structured
)

__all__ = [
    'VetoResolutionSummary',
    'BatchVetoSummaries', 
    'LLMProcessingResult',
    'EnhancedVetoRecord',
    'StructuredLLMClient',
    'create_veto_summary_client',
    'generate_veto_summary_structured'
]


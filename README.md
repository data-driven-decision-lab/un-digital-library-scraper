# UN Digital Library Scraper

## Project Overview
Pipeline for scraping, processing, and analyzing UN resolution voting data from the UN Digital Library. This tool combines web scraping capabilities with advanced machine learning-based tagging to classify resolutions by both subject matter and geographical relevance.

## Key Features
- Automated scraping of UN resolution voting data
- Incremental data collection to avoid duplicate processing
- Subject matter classification using OpenAI's API
- Enhanced geographical tagging with both pattern matching and LLM approaches
- Extraction of country, subregion, and continent information



## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/un-digital-library-scraper.git
cd un-digital-library-scraper

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to add your OpenAI API key
```

## Quick Start

```bash
# Run the complete pipeline
python pipeline_complete.py

## Project Structure

```

## Project Structure


* pipeline_complete.py                            # Main script integrating all functionality
* README.md                                       # This documentation
* requirements.txt                                # Python dependencies
* .env.example                                    # Template for environment variables
* dictionaries/                                   # Reference data, e.g un and geographic classifiers
   - un_classification.py                         # Classification scheme
   - un_geo_hierarchy.py                          # Geographical hierarchy
   - iso2_country.py                              # ISO country mappings
* pipeline_output/                                # Output CSV files
   - UN_VOTING_DATA_RAW_WITH_TAGS_YYYY-MM-DD.csv  # Example output file
* .gitignore                                      # Make sure your .env is present     


## API Details

The project includes a FastAPI-based API to serve the processed data and analysis.

### Country Report Endpoint

*   **Endpoint:** `/report/{country_iso}`
*   **Method:** `GET`
*   **Path Parameters:**
    *   `country_iso` (string, required): The 3-letter uppercase ISO code of the country (e.g., "USA").
*   **Query Parameters:**
    *   `start_year` (integer, required): The start year for the report period.
    *   `end_year` (integer, required): The end year for the report period.
*   **Description:** Provides a detailed report on a country's UN voting patterns, scores, and regional context for a specified period.
*   **Response Structure:** The JSON response includes several sections:
    *   `report_metadata`: Basic information about the report (country ISO, name, period).
    *   `world_average_scores_period`: Average pillar scores and total index average for the world during the period.
    *   `country_average_scores_period`: Average pillar scores, ranks, and total index average for the selected country over the period.
    *   `index_score_analysis`: The selected country's index score and rank at the start and end of the period, and percentage change.
    *   `voting_behavior_overall`: The selected country's voting statistics (yes, no, abstain percentages) compared to world averages for the period.
    *   `most_aligned_p5_member`: Details of the UN Security Council P5 member whose voting record (based on cosine similarity for the period) is most aligned with the selected country. Includes the P5 country's ISO, name, and scaled similarity score.
    *   `least_aligned_p5_member`: Details of the UN Security Council P5 member whose voting record is least aligned with the selected country. Includes the P5 country's ISO, name, and scaled similarity score.
    *   `scores_timeseries`: A yearly breakdown of the selected country's pillar scores, ranks, and total index average, alongside yearly world averages for these scores.
    *   `top_allies`: Top 5 countries most aligned with the selected country based on average voting similarity over the period.
    *   `top_enemies`: Top 5 countries least aligned with the selected country based on average voting similarity over the period.
    *   `top_supported_topics`: Top 3 topics most supported by the selected country.
    *   `top_opposed_topics`: Top 3 topics most opposed by the selected country.
    *   `all_topic_voting`: Detailed voting statistics for the selected country across all topics compared to world averages.
    *   `regional_context`: Provides context about the selected country's voting alignment within its UN-defined geographical region for the `end_year` of the reporting period.
        *   `un_region` (string): The UN region name of the selected country.
        *   `regional_peer_alignment` (list of objects): A list of other countries in the same UN region. Each object contains:
            *   `country_iso3` (string): ISO code of the peer country.
            *   `country_name` (string): Full name of the peer country.
            *   `alignment_score` (float): The scaled cosine similarity score (0-100) indicating alignment with the *selected country* for the `end_year`. Sorted by alignment score in descending order.
        *   `average_regional_alignment_score` (float): The average of the alignment scores between the selected country and all its regional peers for the `end_year`.

## API Key Configuration

This project requires an OpenAI API key to function properly. The API key is used for the classification and geo-tagging features that analyze resolution titles.

### Setting Up Your API Key

1. Create an OpenAI Account: If you don't already have an API key, sign up at OpenAI's platform

2. Create a .env File: In the root directory of the project, create a file named .env

3. Add Your API Key: Add the following line to your .env file:
   API_KEY=your_openai_api_key_here

### API Usage Considerations

- The pipeline uses GPT-4o-mini by default, which offers a good balance of accuracy and cost
- API calls are made for each resolution title during tagging
- Costs will vary depending on the volume of resolutions processed

### Security Notes

- NEVER commit your .env file to version control, make sure the .env file is included in .gitignore!

## Data Output

### File Naming Convention
The pipeline generates CSV files with the following naming pattern:

UN_VOTING_DATA_RAW_WITH_TAGS_YYYY-MM-DD.csv

Where YYYY-MM-DD is the generation date (e.g., UN_VOTING_DATA_RAW_WITH_TAGS_2025-03-25.csv).

### Data Schema

- id: Unique record identifier (e.g., 42)
- Council: Issuing UN council (e.g., Security Council)
- Date: Resolution voting date (e.g., 2020-06-15)
- Title: Resolution title (e.g., The situation in the Middle East)
- Resolution: Resolution ID (e.g., S/RES/2532 (2020))
- country: Relevant countries (e.g., SY, IL, LB)
- subregion: Relevant subregions (e.g., Western Asia, Southern Europe)
- continent: Relevant continents (e.g., Asia, Europe)
- tags: Subject matter tags (e.g., PEACE AND SECURITY, HUMANITARIAN AID)
- TOTAL VOTES: Total votes cast (e.g., 15)
- YES COUNT: Number of yes votes (e.g., 13)
- NO COUNT: Number of no votes (e.g., 0)
- ABSTENTION COUNT: Number of abstentions (e.g., 2)
- Link: Source URL (e.g., https://digitallibrary.un.org/record/3870984)
- Per country voting data... YES, NO, ABSTAIN, or blank

### Sample Data

id,"Security Council","2020-07-11","The situation in Syria","S/RES/2533 (2020)","SY","Western Asia","Asia","PEACE AND SECURITY, HUMANITARIAN AID",15,12,0,3, (followed by individual country voting data, YES, NO, ABSTAIN, ' ')

## Methodology

### Classification Approach

The project employs a carefully designed approach to categorize UN resolutions by both subject matter and geographic relevance:

#### Tags Classification System

The subject matter tagging system is built upon the UNBIS Thesaurus (United Nations Bibliographic Information System Thesaurus), available at https://metadata.un.org/thesaurus/?lang=en. This standardized thesaurus was chosen for the following reasons:

- Official UN Standard: Using the UN's own classification system ensures alignment with how the organization itself categorizes documents
- Hierarchical Structure: The UNBIS Thesaurus provides a well-organized multi-level hierarchy that allows for increasingly specific categorization
- Comprehensive Coverage: The thesaurus covers the full breadth of UN activities and interests

The classification pipeline is designed to identify and apply these standardized tags at three levels:
1. Main Category: Broad thematic areas (e.g., PEACE AND SECURITY, ECONOMIC DEVELOPMENT)
2. Subcategory: More specific topics within each main category
3. Specific Items: Detailed descriptors for precise categorization

A resolution can have multiple tags, in no case can it have a Specific Item without a Subcategory or Main Category, likewise a Subcategory cannot exist without a Main.

We leverage OpenAI's GPT-4o-mini model to analyze resolution titles and determine the appropriate tags from the UNBIS hierarchy.

#### Geographic Classification System

The geographic tagging system also follows the UNBIS Thesaurus structure for "Geographic Descriptors" (https://metadata.un.org/thesaurus/17?lang=en) and implements a dual-method approach for maximum accuracy:

1. Pattern Matching (Primary Method):
   - Uses regex to identify explicit mentions of countries, regions, and continents
   - Hierarchical matching (searches for countries first, then regions, then continents)
   - Prioritizes longer matches to avoid ambiguity (e.g., "New Zealand" vs just "Zealand")

2. LLM Analysis (Verification & Enhancement):
   - Utilizes GPT-4o-mini to catch implicit geographic references
   - Identifies locations that might be mentioned by context rather than direct naming
   - Helps with cases where locations are referenced by alternative names, political entities, or local regions

3. Standardization:
   - Countries are converted to ISO 3166-1 alpha-2 codes for clarity and consistency
   - Maintains the three-level geographic hierarchy from the UNBIS system:
     * Continent: Major world regions (Africa, Americas, Asia, Europe, Oceania)
     * Subregion: Geographic subdivisions within continents
     * Country: Individual nations identified by ISO codes where possible
          Note: The exception here are bodies of water which keep their original name

## Contributing

Contributions are welcome! Here's how you can help:

1. Report bugs - Open an issue if you find any problems
2. Suggest features - Have ideas for improvements? Let us know
3. Feel free to submit pull requests too!

Reach out to the team at "hello@datadrivendecisionlab.com"

## Use Cases

- Academic Research - Analyze voting patterns in UN resolutions
- Policy Analysis - Examine country positions on specific issues
- Diplomatic Studies - Track changes in international relations over time
- Data Journalism - Create visualizations of international consensus/disagreement

## Changelog

### Recent Changes
- 2025-03-26: Integrated enhanced geo-tagging functionality with regex and LLM approaches

## Dependencies

Main dependencies:
- openai: LLM API access for classification
- pandas: Data manipulation and analysis
- selenium: Web scraping automation
- beautifulsoup4: HTML parsing
- pydantic: Data validation
- numpy: Numerical operations
- tqdm: Progress indication
- python-dotenv: Environment variable management

## Limitations and Ethical Considerations

- The scraper relies on the UN Digital Library's structure, which may change
- Rate limiting may affect data collection speed
- OpenAI API usage incurs costs based on token usage
- Not all resolutions have complete voting data
- Please attribute when using this data in publications


## Citations

If you use this tool or data in your research, please cite:

@software{un_digital_library_scraper,
  author = {3DL},
  title = {UN Digital Library Scraper},
  year = {2025},
  url = {https://github.com/yourusername/un-digital-library-scraper}
}

## License

MIT License

Copyright (c) 2025 3DL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

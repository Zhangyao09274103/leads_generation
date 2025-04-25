# AI-Powered Lead Generation System

This project implements an intelligent lead generation system for identifying and analyzing potential small business customers. It uses AI and machine learning to scrape, analyze, and prioritize business leads based on various data points from Google Maps and business websites.

## Features

1. **Automated Data Collection**
   - Scrapes business information from Google Maps
   - Extracts contact details, ratings, and reviews
   - Collects website content when available

2. **AI-Powered Analysis**
   - Uses LLM for business summary generation
   - Performs sentiment analysis on reviews
   - Predicts potential revenue using machine learning
   - Prioritizes leads based on multiple factors

3. **Visualization and Reporting**
   - Interactive scatter plots of lead data
   - Exportable CSV reports
   - Detailed business summaries

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Install Chrome WebDriver:
   - Download ChromeDriver matching your Chrome version
   - Add it to your system PATH

## Usage

Run the main script:
```bash
python lead_generation.py
```

The script will:
1. Collect business data from Google Maps
2. Analyze each business using AI
3. Generate revenue predictions
4. Prioritize leads
5. Create visualizations
6. Export results to `high_quality_leads.csv`

## Output

The system generates:
1. A CSV file containing:
   - Business names and contact information
   - Predicted revenue
   - Ratings and review counts
   - AI-generated business summaries

2. Interactive visualization showing:
   - Review count vs. Predicted revenue
   - Rating indicated by point size
   - Hover data for additional details

## Customization

Modify the following parameters in `lead_generation.py`:
- `location`: Target geographic area
- `business_type`: Type of businesses to search for
- Scoring weights in `LeadPrioritizationAgent`
- Revenue prediction features in `RevenuePredictionAgent`

## Notes

- The revenue prediction model uses simulated data for demonstration
- In a production environment, replace with real historical data
- Respect rate limits and terms of service when scraping
- Consider implementing proxy rotation for large-scale scraping

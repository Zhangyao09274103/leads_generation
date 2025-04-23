import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain.agents import Tool, AgentExecutor
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Data structures
@dataclass
class BusinessLead:
    business_name: str
    address: str
    phone: Optional[str]
    website: Optional[str]
    review_count: int
    rating: float
    reviews: List[str]
    business_type: str
    extracted_summary: Optional[str]
    predicted_revenue: Optional[float]

class GoogleMapsScraperAgent:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def scrape_business_data(self, location: str, business_type: str) -> List[BusinessLead]:
        search_query = f"{business_type} in {location}"
        url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"
        
        self.driver.get(url)
        leads = []
        
        # Wait for results to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "section-result"))
        )
        
        # Extract business listings
        business_elements = self.driver.find_elements(By.CLASS_NAME, "section-result")
        
        for element in business_elements[:10]:  # Limit to first 10 results for demo
            try:
                name = element.find_element(By.CLASS_NAME, "section-result-title").text
                address = element.find_element(By.CLASS_NAME, "section-result-location").text
                rating = float(element.find_element(By.CLASS_NAME, "section-result-rating").text)
                review_count = int(element.find_element(By.CLASS_NAME, "section-result-review-count")
                                 .text.replace("(", "").replace(")", ""))
                
                # Click to open business details
                element.click()
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "section-info-container"))
                )
                
                # Get additional details
                details = self.driver.find_element(By.CLASS_NAME, "section-info-container")
                phone = details.find_element(By.CLASS_NAME, "section-info-phone").text
                website = details.find_element(By.CLASS_NAME, "section-info-website").text
                
                # Get reviews
                reviews = []
                review_elements = details.find_elements(By.CLASS_NAME, "section-review-text")
                for review in review_elements[:5]:  # Get first 5 reviews
                    reviews.append(review.text)
                
                lead = BusinessLead(
                    business_name=name,
                    address=address,
                    phone=phone,
                    website=website,
                    review_count=review_count,
                    rating=rating,
                    reviews=reviews,
                    business_type=business_type,
                    extracted_summary=None,
                    predicted_revenue=None
                )
                leads.append(lead)
                
            except Exception as e:
                print(f"Error processing business: {str(e)}")
                continue
        
        return leads

class WebsiteScraperAgent:
    def scrape_website(self, url: str) -> str:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = []
            for tag in ['p', 'h1', 'h2', 'h3', 'li']:
                elements = soup.find_all(tag)
                content.extend([elem.text.strip() for elem in elements])
            
            return " ".join(content)
        except Exception as e:
            print(f"Error scraping website {url}: {str(e)}")
            return ""

class LLMAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def analyze_business(self, lead: BusinessLead) -> BusinessLead:
        # Analyze business information using LLM
        prompt = f"""
        Analyze the following business information and provide a detailed summary:
        Business Name: {lead.business_name}
        Business Type: {lead.business_type}
        Reviews: {lead.reviews[:5]}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        lead.extracted_summary = response.content
        
        # Analyze review sentiment
        sentiments = [
            self.sentiment_analyzer(review)[0]
            for review in lead.reviews
        ]
        
        # Add sentiment scores to summary
        positive_reviews = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        sentiment_ratio = positive_reviews / len(sentiments) if sentiments else 0
        lead.extracted_summary += f"\nSentiment Analysis: {sentiment_ratio:.2%} positive reviews"
        
        return lead

class RevenuePredictionAgent:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train_model(self, training_data: List[BusinessLead]):
        # In a real implementation, you would have historical data
        # For demo purposes, we'll create a simple model based on available features
        X = []
        y = []
        
        for lead in training_data:
            features = [
                lead.review_count,
                lead.rating,
                1 if lead.website else 0,
                len(lead.reviews)
            ]
            X.append(features)
            
            # Simulate revenue based on features (for demo)
            simulated_revenue = (
                lead.review_count * 1000 +
                lead.rating * 50000 +
                (100000 if lead.website else 0)
            )
            y.append(simulated_revenue)
        
        self.model.fit(X, y)
    
    def predict_revenue(self, lead: BusinessLead) -> float:
        features = [
            lead.review_count,
            lead.rating,
            1 if lead.website else 0,
            len(lead.reviews)
        ]
        
        predicted_revenue = self.model.predict([features])[0]
        return predicted_revenue

class LeadPrioritizationAgent:
    def prioritize_leads(self, leads: List[BusinessLead]) -> List[BusinessLead]:
        # Score leads based on multiple factors
        scored_leads = []
        for lead in leads:
            score = self._calculate_lead_score(lead)
            scored_leads.append((score, lead))
        
        scored_leads.sort(reverse=True)
        return [lead for _, lead in scored_leads]
    
    def _calculate_lead_score(self, lead: BusinessLead) -> float:
        # Calculate a composite score based on various factors
        score = 0.0
        
        # Revenue potential (40% weight)
        if lead.predicted_revenue:
            score += 0.4 * (min(lead.predicted_revenue / 1000000, 1.0))
        
        # Rating score (30% weight)
        score += 0.3 * (lead.rating / 5.0)
        
        # Review count score (20% weight)
        review_score = min(lead.review_count / 100, 1.0)
        score += 0.2 * review_score
        
        # Website presence (10% weight)
        if lead.website:
            score += 0.1
        
        return score

def visualize_results(leads: List[BusinessLead]):
    # Create DataFrame for visualization
    df = pd.DataFrame([
        {
            'business_name': lead.business_name,
            'predicted_revenue': lead.predicted_revenue,
            'review_count': lead.review_count,
            'rating': lead.rating
        }
        for lead in leads
    ])
    
    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='review_count',
        y='predicted_revenue',
        size='rating',
        hover_data=['business_name'],
        title='Business Leads Analysis'
    )
    
    return fig

def main():
    # Initialize agents
    maps_scraper = GoogleMapsScraperAgent()
    website_scraper = WebsiteScraperAgent()
    llm_analyzer = LLMAnalysisAgent()
    revenue_predictor = RevenuePredictionAgent()
    lead_prioritizer = LeadPrioritizationAgent()
    
    # Set search parameters
    location = "San Francisco, CA"
    business_type = "small business"
    
    # 1. Collect leads from Google Maps
    print("Collecting leads from Google Maps...")
    leads = maps_scraper.scrape_business_data(location, business_type)
    
    # 2. Enrich data with website content
    print("Enriching data with website content...")
    for lead in leads:
        if lead.website:
            website_content = website_scraper.scrape_website(lead.website)
            lead.website_content = website_content
    
    # 3. Analyze leads using LLM
    print("Analyzing leads with LLM...")
    analyzed_leads = [llm_analyzer.analyze_business(lead) for lead in leads]
    
    # 4. Train and predict revenue
    print("Predicting revenue...")
    revenue_predictor.train_model(analyzed_leads)  # In real implementation, use historical data
    for lead in analyzed_leads:
        lead.predicted_revenue = revenue_predictor.predict_revenue(lead)
    
    # 5. Prioritize leads
    print("Prioritizing leads...")
    prioritized_leads = lead_prioritizer.prioritize_leads(analyzed_leads)
    
    # 6. Visualize results
    print("Generating visualization...")
    fig = visualize_results(prioritized_leads)
    fig.show()
    
    # 7. Export results
    results_df = pd.DataFrame([
        {
            'business_name': lead.business_name,
            'contact': lead.phone,
            'predicted_revenue': lead.predicted_revenue,
            'rating': lead.rating,
            'review_count': lead.review_count,
            'summary': lead.extracted_summary
        }
        for lead in prioritized_leads
    ])
    
    results_df.to_csv('high_quality_leads.csv', index=False)
    print("Results exported to high_quality_leads.csv")

if __name__ == "__main__":
    main() 
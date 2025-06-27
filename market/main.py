
import pandas as pd
import numpy as np
import feedparser
import requests
from datetime import datetime, timedelta
import time
import re
from collections import Counter
import json
import base64
from io import BytesIO
import os

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# NLP libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

class NewsDataCollector:
    """Handles news data collection from various RSS sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_google_news_url(self, query, region='US', language='en', category=None):
        """Generate Google News RSS URL with parameters"""
        base_url = "https://news.google.com/rss"
        
        if category:
            url = f"{base_url}/headlines/section/topic/{category}?hl={language}&gl={region}&ceid={region}:{language}"
        elif query:
            encoded_query = requests.utils.quote(query)
            url = f"{base_url}/search?q={encoded_query}&hl={language}&gl={region}&ceid={region}:{language}"
        else:
            url = f"{base_url}?hl={language}&gl={region}&ceid={region}:{language}"
            
        return url
    
    def scrape_rss_feed(self, url, max_articles=100):
        """Scrape articles from RSS feed"""
        try:
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown'),
                    'timestamp': datetime.now()
                }
                articles.append(article)
                
            return articles
        except Exception as e:
            st.error(f"Error scraping feed: {str(e)}")
            return []
    
    def collect_news_data(self, query=None, region='US', category=None, max_articles=100):
        """Main method to collect news data"""
        url = self.get_google_news_url(query, region, category=category)
        return self.scrape_rss_feed(url, max_articles)

class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        try:
            # Try to load Hugging Face sentiment model
            self.hf_analyzer = pipeline("sentiment-analysis", 
                                      model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                      return_all_scores=True)
            self.use_hf = True
        except:
            self.use_hf = False
            st.warning("Advanced sentiment model not available, using VADER + TextBlob")
    
    def analyze_sentiment_vader(self, text):
        """VADER sentiment analysis"""
        scores = self.vader.polarity_scores(text)
        return scores
    
    def analyze_sentiment_textblob(self, text):
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def get_sentiment_label(self, compound_score):
        """Convert compound score to descriptive label"""
        if compound_score >= 0.5:
            return "Very Positive"
        elif compound_score >= 0.1:
            return "Positive"
        elif compound_score >= -0.1:
            return "Neutral"
        elif compound_score >= -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def analyze_text(self, text):
        """Comprehensive sentiment analysis"""
        vader_scores = self.analyze_sentiment_vader(text)
        textblob_scores = self.analyze_sentiment_textblob(text)
        
        result = {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_scores['polarity'],
            'textblob_subjectivity': textblob_scores['subjectivity'],
            'sentiment_label': self.get_sentiment_label(vader_scores['compound'])
        }
        
        # Use Hugging Face if available
        if self.use_hf:
            try:
                hf_result = self.hf_analyzer(text[:512])[0]  # Limit text length
                result['hf_sentiment'] = hf_result
            except:
                pass
                
        return result

class TextSummarizer:
    """Text summarization using Hugging Face transformers"""
    
    def __init__(self):
        try:
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn",
                                     max_length=150, 
                                     min_length=30,
                                     do_sample=False)
            self.available = True
        except:
            try:
                # Fallback to smaller model
                self.summarizer = pipeline("summarization",
                                         model="sshleifer/distilbart-cnn-12-6",
                                         max_length=100,
                                         min_length=20)
                self.available = True
            except:
                self.available = False
                st.warning("Summarization model not available")
    
    def summarize_headlines(self, headlines, max_headlines=10):
        """Summarize a list of headlines"""
        if not self.available or not headlines:
            return "Summarization not available"
        
        try:
            # Combine top headlines
            combined_text = ". ".join(headlines[:max_headlines])
            if len(combined_text) < 50:
                return "Insufficient text for summarization"
            
            # Limit text length for model
            if len(combined_text) > 1000:
                combined_text = combined_text[:1000]
            
            summary = self.summarizer(combined_text)[0]['summary_text']
            return summary
        except Exception as e:
            return f"Summarization error: {str(e)}"

class DataVisualizer:
    """Advanced data visualization for insights"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def create_wordcloud(self, text_data, exclude_words=None, colormap='viridis'):
        """Generate customizable word cloud"""
        if exclude_words is None:
            exclude_words = set(['news', 'says', 'new', 'get', 'make', 'take'])
        
        # Clean and combine text
        text = ' '.join(text_data).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove common words
        words = [word for word in text.split() if len(word) > 2 and word not in exclude_words]
        text = ' '.join(words)
        
        if not text.strip():
            return None
            
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        return wordcloud
    
    def plot_sentiment_distribution(self, sentiments):
        """Create sentiment distribution bar chart"""
        sentiment_counts = Counter(sentiments)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                marker_color=['#ff4444', '#ff8800', '#888888', '#44aa44', '#00aa00']
            )
        ])
        
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            template="plotly_white"
        )
        
        return fig
    
    def plot_sentiment_timeline(self, df):
        """Create sentiment trend over time"""
        if 'timestamp' not in df.columns:
            return None
            
        # Group by hour and calculate average sentiment
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_sentiment = df.groupby('hour')['vader_compound'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_sentiment['hour'],
            y=hourly_sentiment['vader_compound'],
            mode='lines+markers',
            name='Average Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Sentiment Trend Over Time",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            template="plotly_white"
        )
        
        return fig

class DataExporter:
    """Handle data export in multiple formats"""
    
    @staticmethod
    def to_csv(df):
        """Export DataFrame to CSV"""
        return df.to_csv(index=False)
    
    @staticmethod
    def to_json(df):
        """Export DataFrame to JSON"""
        return df.to_json(orient='records', date_format='iso')
    
    @staticmethod
    def wordcloud_to_png(wordcloud):
        """Convert wordcloud to PNG bytes"""
        if wordcloud is None:
            return None
            
        img_buffer = BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        return img_buffer.getvalue()
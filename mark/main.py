# MarketPulse Pro - Python 3.13 Compatible Version
# Simplified version that works with newer Python versions


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

# Visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Basic NLP (Python 3.13 compatible)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class NewsDataCollector:
    """Handles news data collection from RSS sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_google_news_url(self, query, region='US', language='en', category=None):
        """Generate Google News RSS URL"""
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

class SimpleSentimentAnalyzer:
    """Simplified sentiment analysis for Python 3.13 compatibility"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Enhanced word lists for better accuracy
        self.positive_words = {
            'excellent', 'amazing', 'outstanding', 'fantastic', 'great', 'good', 'positive', 
            'success', 'growth', 'profit', 'win', 'best', 'improve', 'boost', 'surge', 
            'rise', 'gain', 'advance', 'breakthrough', 'innovation', 'achievement', 'victory',
            'strong', 'robust', 'healthy', 'bullish', 'optimistic', 'confident', 'promising'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'negative', 'loss', 'decline', 'fail', 
            'worst', 'crisis', 'problem', 'risk', 'drop', 'fall', 'crash', 'collapse',
            'struggle', 'challenge', 'concern', 'worry', 'fear', 'threat', 'danger',
            'weak', 'poor', 'bearish', 'pessimistic', 'uncertain', 'volatile', 'unstable'
        }
    
    def enhanced_rule_based_sentiment(self, text):
        """Enhanced rule-based sentiment analysis"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        # Calculate sentiment score
        total_words = len(words)
        if total_words == 0:
            return 0.0
            
        sentiment_score = (pos_count - neg_count) / total_words
        
        # Normalize to -1 to 1 range
        sentiment_score = max(-1, min(1, sentiment_score * 10))
        
        return sentiment_score
    
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
        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # Enhanced rule-based analysis
        enhanced_score = self.enhanced_rule_based_sentiment(text)
        
        # Combine scores (weighted average)
        combined_score = (vader_scores['compound'] * 0.7) + (enhanced_score * 0.3)
        
        result = {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'enhanced_score': enhanced_score,
            'combined_score': combined_score,
            'sentiment_label': self.get_sentiment_label(combined_score)
        }
        
        return result

class SimpleWordCloud:
    """Simple word cloud alternative using matplotlib"""
    
    def create_word_frequency_chart(self, text_data, exclude_words=None, max_words=20):
        """Create word frequency chart instead of word cloud"""
        if exclude_words is None:
            exclude_words = {'news', 'says', 'new', 'get', 'make', 'take', 'will', 'can', 'may'}
        
        # Combine and clean text
        text = ' '.join(text_data).lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Filter words
        filtered_words = [
            word for word in words 
            if len(word) > 3 and word not in exclude_words
        ]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(max_words)
        
        if not top_words:
            return None
            
        # Create bar chart
        words, counts = zip(*top_words)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(words)), counts, color='steelblue')
        
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Frequency')
        ax.set_title('Top Keywords in Headlines')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(counts[i]), ha='left', va='center')
        
        plt.tight_layout()
        return fig

class DataVisualizer:
    """Data visualization for insights"""
    
    def plot_sentiment_distribution(self, sentiments):
        """Create sentiment distribution chart"""
        sentiment_counts = Counter(sentiments)
        
        colors = {
            'Very Positive': '#2E8B57',
            'Positive': '#90EE90', 
            'Neutral': '#808080',
            'Negative': '#FFA500',
            'Very Negative': '#DC143C'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values()),
                marker_color=[colors.get(k, '#808080') for k in sentiment_counts.keys()]
            )
        ])
        
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def plot_sentiment_timeline(self, df):
        """Create sentiment trend over time"""
        if 'timestamp' not in df.columns or len(df) < 2:
            return None
            
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['combined_score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title="Sentiment Trend Over Time",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            template="plotly_white",
            height=400
        )
        
        return fig


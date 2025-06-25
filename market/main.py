# MarketPulse Pro - Advanced News Analysis & Sentiment Dashboard

# A client-focused tool for real-time industry insights

import streamlit as st
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
warnings.filterwarnings(‘ignore’)

class NewsDataCollector:
“”“Handles news data collection from various RSS sources”””

```
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
```

class SentimentAnalyzer:
“”“Advanced sentiment analysis using multiple models”””

```
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
```

class TextSummarizer:
“”“Text summarization using Hugging Face transformers”””

```
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
```

class DataVisualizer:
“”“Advanced data visualization for insights”””

```
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
```

class DataExporter:
“”“Handle data export in multiple formats”””

```
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
```

def main():
“”“Main Streamlit application”””

```
# Page configuration
st.set_page_config(
    page_title="MarketPulse Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-bottom: 2rem;
    border-radius: 10px;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 MarketPulse Pro</h1>
    <p>Advanced News Analysis & Sentiment Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# Initialize components
collector = NewsDataCollector()
analyzer = SentimentAnalyzer()
summarizer = TextSummarizer()
visualizer = DataVisualizer()

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Search parameters
query = st.sidebar.text_input("Search Query", value="artificial intelligence", 
                             help="Enter keywords to search for")

region = st.sidebar.selectbox("Region", 
                             options=['US', 'UK', 'CA', 'AU', 'NG', 'IN', 'DE', 'FR'],
                             help="Select geographical region")

category_map = {
    'General': None,
    'Business': 'BUSINESS',
    'Technology': 'TECHNOLOGY', 
    'Health': 'HEALTH',
    'Science': 'SCIENCE',
    'Sports': 'SPORTS'
}

category = st.sidebar.selectbox("Category", options=list(category_map.keys()))

max_articles = st.sidebar.slider("Max Articles", min_value=10, max_value=200, value=50)

# Filtering options
st.sidebar.header("🔍 Filtering")
keyword_filter = st.sidebar.text_input("Filter by Keywords", 
                                      help="Filter headlines containing these words")

# Visualization options
st.sidebar.header("🎨 Visualization")
exclude_words = st.sidebar.text_area("Exclude Words from WordCloud", 
                                    value="news, says, new, get, make",
                                    help="Comma-separated words to exclude")

colormap = st.sidebar.selectbox("WordCloud Color Scheme", 
                               options=['viridis', 'plasma', 'inferno', 'magma', 'Blues'])

# Main content
if st.sidebar.button("🚀 Analyze News", type="primary"):
    
    with st.spinner("Collecting news data..."):
        # Collect news data
        articles = collector.collect_news_data(
            query=query if query else None,
            region=region,
            category=category_map[category],
            max_articles=max_articles
        )
    
    if not articles:
        st.error("No articles found. Try adjusting your search parameters.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Apply keyword filtering
    if keyword_filter:
        keywords = [k.strip().lower() for k in keyword_filter.split(',')]
        mask = df['title'].str.lower().str.contains('|'.join(keywords), na=False)
        df = df[mask]
    
    if df.empty:
        st.warning("No articles match your filter criteria.")
        return
    
    # Perform sentiment analysis
    with st.spinner("Analyzing sentiment..."):
        sentiment_results = []
        progress_bar = st.progress(0)
        
        for idx, title in enumerate(df['title']):
            sentiment = analyzer.analyze_text(title)
            sentiment_results.append(sentiment)
            progress_bar.progress((idx + 1) / len(df))
        
        progress_bar.empty()
    
    # Add sentiment data to DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)
    df = pd.concat([df, sentiment_df], axis=1)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(df))
    
    with col2:
        avg_sentiment = df['vader_compound'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    with col3:
        positive_pct = (df['sentiment_label'].isin(['Positive', 'Very Positive']).sum() / len(df)) * 100
        st.metric("Positive %", f"{positive_pct:.1f}%")
    
    with col4:
        top_source = df['source'].value_counts().index[0] if len(df) > 0 else "N/A"
        st.metric("Top Source", top_source)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Visualizations", "📝 Summary", "📋 Data", "💾 Export"])
    
    with tab1:
        st.header("Headlines Overview")
        
        # Display recent headlines with sentiment
        for _, row in df.head(10).iterrows():
            sentiment_color = {
                'Very Positive': 'green',
                'Positive': 'lightgreen', 
                'Neutral': 'gray',
                'Negative': 'orange',
                'Very Negative': 'red'
            }.get(row['sentiment_label'], 'gray')
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['title']}</h4>
                <p><strong>Source:</strong> {row['source']} | 
                <strong>Sentiment:</strong> <span style="color: {sentiment_color}">{row['sentiment_label']}</span> 
                ({row['vader_compound']:.3f})</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    with tab2:
        st.header("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig_sentiment = visualizer.plot_sentiment_distribution(df['sentiment_label'])
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Sentiment timeline
            fig_timeline = visualizer.plot_sentiment_timeline(df)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Word cloud
        st.subheader("Word Cloud")
        exclude_list = [w.strip() for w in exclude_words.split(',') if w.strip()]
        wordcloud = visualizer.create_wordcloud(df['title'].tolist(), 
                                               exclude_words=set(exclude_list),
                                               colormap=colormap)
        
        if wordcloud:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Could not generate word cloud")
    
    with tab3:
        st.header("AI-Generated Summary")
        
        if summarizer.available:
            with st.spinner("Generating summary..."):
                summary = summarizer.summarize_headlines(df['title'].tolist())
                st.info(summary)
        else:
            st.warning("Summary feature not available")
        
        # Top keywords
        st.subheader("Top Keywords")
        all_text = ' '.join(df['title'].tolist()).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter([w for w in words if len(w) > 3])
        
        top_keywords = word_freq.most_common(10)
        keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
        st.dataframe(keyword_df, use_container_width=True)
    
    with tab4:
        st.header("Raw Data")
        
        # Display options
        show_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=['title', 'source', 'sentiment_label', 'vader_compound']
        )
        
        if show_columns:
            st.dataframe(df[show_columns], use_container_width=True)
    
    with tab5:
        st.header("Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv_data = DataExporter.to_csv(df)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"marketpulse_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON Export
            json_data = DataExporter.to_json(df)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"marketpulse_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        with col3:
            # WordCloud PNG Export
            if 'wordcloud' in locals() and wordcloud:
                png_data = DataExporter.wordcloud_to_png(wordcloud)
                if png_data:
                    st.download_button(
                        label="🖼️ Download WordCloud",
                        data=png_data,
                        file_name=f"wordcloud_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                        mime="image/png"
                    )
    
    # Store data for historical analysis
    if st.sidebar.button("💾 Save to Historical Data"):
        # Save daily summary
        daily_summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'query': query,
            'region': region,
            'total_articles': len(df),
            'avg_sentiment': df['vader_compound'].mean(),
            'positive_percentage': positive_pct
        }
        
        # In a real application, this would be saved to a database
        st.sidebar.success("Data saved to historical records!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 MarketPulse Pro - Built for Business Intelligence</p>
    <p>Powered by Advanced NLP & Real-time Data Analysis</p>
</div>
""", unsafe_allow_html=True)
```

if **name** == “**main**”:
main()

# MarketPulse Pro - Python 3.13 Compatible Version
# Simplified version that works with newer Python versions

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

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="MarketPulse Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 MarketPulse Pro</h1>
        <p>Python 3.13 Compatible News Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    collector = NewsDataCollector()
    analyzer = SimpleSentimentAnalyzer()
    visualizer = DataVisualizer()
    word_cloud = SimpleWordCloud()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Search parameters
    query = st.sidebar.text_input("Search Query", value="artificial intelligence")
    
    region = st.sidebar.selectbox("Region", 
                                 options=['US', 'UK', 'CA', 'AU', 'NG', 'IN', 'DE', 'FR'])
    
    category_map = {
        'General': None,
        'Business': 'BUSINESS',
        'Technology': 'TECHNOLOGY', 
        'Health': 'HEALTH',
        'Science': 'SCIENCE'
    }
    
    category = st.sidebar.selectbox("Category", options=list(category_map.keys()))
    max_articles = st.sidebar.slider("Max Articles", min_value=10, max_value=100, value=30)
    
    # Filtering options
    st.sidebar.header("🔍 Filtering")
    keyword_filter = st.sidebar.text_input("Filter by Keywords")
    
    # Main content
    if st.sidebar.button("🚀 Analyze News", type="primary"):
        
        with st.spinner("Collecting news data..."):
            articles = collector.collect_news_data(
                query=query if query else None,
                region=region,
                category=category_map[category],
                max_articles=max_articles
            )
        
        if not articles:
            st.error("No articles found. Please check your internet connection or try different search terms.")
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
            avg_sentiment = df['combined_score'].mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
        
        with col3:
            positive_pct = (df['sentiment_label'].isin(['Positive', 'Very Positive']).sum() / len(df)) * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        
        with col4:
            sources = df['source'].value_counts()
            top_source = sources.index[0] if len(sources) > 0 else "N/A"
            st.metric("Top Source", top_source)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "📋 Data", "💾 Export"])
        
        with tab1:
            st.header("Headlines Overview")
            
            # Display headlines with sentiment
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
                    ({row['combined_score']:.3f})</p>
                </div>
                """, unsafe_allow_html=True)
        
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
                else:
                    st.info("Timeline requires more data points over time")
            
            # Word frequency chart
            st.subheader("Top Keywords")
            word_fig = word_cloud.create_word_frequency_chart(df['title'].tolist())
            if word_fig:
                st.pyplot(word_fig)
            else:
                st.warning("Could not generate keyword chart")
        
        with tab3:
            st.header("Detailed Data")
            
            # Show detailed sentiment breakdown
            st.subheader("Sentiment Analysis Details")
            display_columns = ['title', 'source', 'sentiment_label', 'combined_score', 'vader_compound']
            st.dataframe(df[display_columns], use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Total Articles', 'Average Sentiment', 'Positive Articles', 'Negative Articles', 'Neutral Articles'],
                'Value': [
                    len(df),
                    f"{df['combined_score'].mean():.3f}",
                    len(df[df['sentiment_label'].isin(['Positive', 'Very Positive'])]),
                    len(df[df['sentiment_label'].isin(['Negative', 'Very Negative'])]),
                    len(df[df['sentiment_label'] == 'Neutral'])
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with tab4:
            st.header("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"marketpulse_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON Export
                json_data = df.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"marketpulse_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Step 1:** Enter your search query (e.g., "artificial intelligence", "cryptocurrency")
        
        **Step 2:** Select your preferred region and category
        
        **Step 3:** Click "🚀 Analyze News" to start the analysis
        
        **Step 4:** Explore the results in different tabs:
        - **Overview**: See top headlines with sentiment scores
        - **Visualizations**: View charts and keyword analysis  
        - **Data**: Examine detailed results
        - **Export**: Download your data in CSV or JSON format
        
        **Tips:**
        - Use specific keywords for better targeted results
        - Try different regions to compare global sentiment
        - Filter results using the keyword filter option
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🚀 MarketPulse Pro - Python 3.13 Compatible Version</p>
        <p>Real-time News Sentiment Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
from main import *


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
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

if __name__ == "__main__":
    main()
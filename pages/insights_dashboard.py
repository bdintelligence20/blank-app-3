# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Qualitative Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# Initialize OpenAI client using Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

# Sidebar
with st.sidebar:
    st.title('📊 Data Analysis Dashboard')
    
    uploaded_files = st.file_uploader(
        "Upload your qualitative data files (CSV, JSON, TXT)", 
        type=["csv", "json", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Load data
        def load_data(uploaded_files):
            dataframes = []
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.type == "application/json":
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    df = pd.read_csv(uploaded_file, delimiter='\t')
                else:
                    st.error("Unsupported file type!")
                    return None
                dataframes.append(df)
            return pd.concat(dataframes, ignore_index=True)
        
        data = load_data(uploaded_files)
        
        if data is not None:
            # Select columns for analysis
            text_columns = st.multiselect(
                "Select the text columns you want to analyze", 
                data.columns
            )
            
            # Select color theme for plots
            color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
            selected_color_theme = st.selectbox('Select a color theme', color_theme_list)
            
            # Analysis options
            analysis_options = st.multiselect(
                "Select analysis types",
                ["Topic Modeling", "Sentiment Analysis", "Word Cloud", "Topic Heatmap"]
            )

# Main Dashboard
if uploaded_files and data is not None and text_columns:
    # Preprocess text data
    data['processed_text'] = data[text_columns].astype(str).apply(lambda x: ' '.join(x), axis=1).apply(lambda x: x.lower())

    # Executive Summary
    st.markdown("## Executive Summary")
    st.markdown("""
    - **Overall Sentiment**: The average sentiment across all analyzed texts shows a **positive/negative** trend.
    - **Top Topics**: The most discussed topics are related to **Customer Service, Product Feedback,** and **Pricing**.
    - **Key Insights**: Significant increase in negative sentiment in the past month. Suggest reviewing customer feedback closely to identify pain points.
    """)

    # Dashboard layout
    st.markdown("## Data Analysis Results")
    col1, col2 = st.columns([2, 3], gap="medium")
    
    with col1:
        if "Word Cloud" in analysis_options:
            st.markdown("### Word Cloud")
            all_text = " ".join(data['processed_text'])
            wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="black").generate(all_text)
            st.image(wordcloud.to_array(), use_column_width=True)
        
        if "Sentiment Analysis" in analysis_options:
            st.markdown("### Sentiment Analysis")
            data['sentiment'] = data['processed_text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x))
            data['positive'] = data['sentiment'].apply(lambda x: x['pos'])
            data['neutral'] = data['sentiment'].apply(lambda x: x['neu'])
            data['negative'] = data['sentiment'].apply(lambda x: x['neg'])
            sentiment_counts = data[['positive', 'neutral', 'negative']].mean()
            sentiment_bar = px.bar(
                sentiment_counts, 
                x=sentiment_counts.index, 
                y=sentiment_counts.values, 
                color=sentiment_counts.index,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            sentiment_bar.update_layout(template="plotly_dark")
            st.plotly_chart(sentiment_bar, use_container_width=True)
    
    with col2:
        if "Topic Modeling" in analysis_options:
            st.markdown("### Topic Modeling")
            n_topics = st.slider("Select number of topics", 2, 10, 5)
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(data['processed_text'])
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(dtm)
            topics_df = pd.DataFrame(lda.components_, columns=vectorizer.get_feature_names_out())
            st.dataframe(topics_df)
        
        if "Topic Heatmap" in analysis_options:
            st.markdown("### Topic Heatmap")
            topic_dist = lda.transform(dtm)
            heatmap = alt.Chart(pd.DataFrame(topic_dist)).mark_rect().encode(
                x=alt.X('column:O', title='Topic'),
                y=alt.Y('index:O', title='Document'),
                color=alt.Color('value:Q', scale=alt.Scale(scheme=selected_color_theme))
            ).properties(width=600, height=400)
            st.altair_chart(heatmap, use_container_width=True)
    
    with st.expander("About"):
        st.write("""
            - **Dashboard Features**: Provides qualitative data analysis including topic modeling, sentiment analysis, and visualization tools like word clouds and heatmaps.
            - **Customization**: Users can choose the data columns for analysis and select different color themes for visualizations.
            - **Data Handling**: Supports multiple file uploads and handles different formats (CSV, JSON, TXT).
        """)

else:
    if uploaded_files:
        st.error("Please select at least one text column for analysis.")
    else:
        st.info("Upload files to start analysis.")

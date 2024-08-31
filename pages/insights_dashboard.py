# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
from sklearn.manifold import TSNE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

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
                ["Topic Modeling", "Sentiment Analysis", "Word Cloud", "Topic Clustering"]
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
            
            # Clustering Topics
            st.markdown("### Topic Clustering")
            kmeans = KMeans(n_clusters=n_topics, random_state=42)
            topics_clustered = kmeans.fit_predict(lda.components_)
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(lda.components_)
            cluster_df = pd.DataFrame(tsne_results, columns=['X', 'Y'])
            cluster_df['Cluster'] = topics_clustered
            
            # Scatter plot for clusters
            scatter_plot = px.scatter(cluster_df, x='X', y='Y', color='Cluster', color_continuous_scale=selected_color_theme)
            scatter_plot.update_layout(template="plotly_dark")
            st.plotly_chart(scatter_plot, use_container_width=True)
            
            # LLM Interpretation for Each Cluster
            st.markdown("### LLM Interpretations of Clusters")
            interpretations = []
            for cluster_num in range(n_topics):
                cluster_text = " ".join(vectorizer.get_feature_names_out()[topics_df.iloc[cluster_num].argsort()[-10:]])
                interpretation_prompt = f"Interpret the following cluster of topics: {cluster_text}"
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a data analyst."},
                              {"role": "user", "content": interpretation_prompt}],
                    temperature=1,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                interpretation = response.choices[0].message.content
                interpretations.append(f"**Cluster {cluster_num+1}**: {interpretation}")
            
            for interpretation in interpretations:
                st.markdown(interpretation)
    
    with st.expander("About"):
        st.write("""
            - **Dashboard Features**: Provides qualitative data analysis including topic modeling, sentiment analysis, clustering of topics, and visualization tools like word clouds and scatter plots.
            - **Customization**: Users can choose the data columns for analysis and select different color themes for visualizations.
            - **Data Handling**: Supports multiple file uploads and handles different formats (CSV, JSON, TXT).
        """)

else:
    if uploaded_files:
        st.error("Please select at least one text column for analysis.")
    else:
        st.info("Upload files to start analysis.")

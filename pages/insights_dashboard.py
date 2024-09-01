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
import os
import PyPDF2
from io import StringIO
import base64

# Set up page configuration with a modern layout
st.set_page_config(
    page_title="Advanced Qualitative Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
        .sidebar .sidebar-content { background-color: #2A3E4C; }
        .css-18e3th9 { padding: 1rem; }
        .css-1lcbmhc { font-size: 1rem; font-weight: bold; }
        .stButton > button { border-radius: 5px; }
        .css-ffhzg2 { font-size: 16px; font-weight: bold; }
        .css-1d391kg { border: 1px solid #2A3E4C; }
        .stDataFrame { border-radius: 8px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# Function to read markdown files
def read_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Function to read PDF files
def read_pdf_file(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Ensuring no NoneType error
    return text

# Function to read text file containing potential keywords
def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().splitlines()
    except FileNotFoundError:
        st.error(f"The file {filepath} was not found.")
        return []

# Function to extract context from files in the "knowledge" folder
def extract_context_from_knowledge():
    context = ""
    knowledge_folder_path = "knowledge"
    
    if os.path.exists(knowledge_folder_path):
        for filename in os.listdir(knowledge_folder_path):
            filepath = os.path.join(knowledge_folder_path, filename)
            if filename.endswith('.md'):
                context += read_markdown_file(filepath)
            elif filename.endswith('.pdf'):
                context += read_pdf_file(filepath)
    return context

# Load current keyword data from the "current_keywords" folder
def load_current_keyword_data():
    current_keywords_folder = os.path.join(os.getcwd(), "current_keywords")
    current_keywords_data = None

    # Check if the folder exists
    if os.path.exists(current_keywords_folder):
        st.write(f"Found current_keywords folder at: {current_keywords_folder}")  # Debug statement
        # List all CSV files in the folder
        files = [f for f in os.listdir(current_keywords_folder) if f.endswith('.csv')]
        if files:
            st.write(f"CSV files found: {files}")  # Debug statement
            # Pick the first CSV file
            current_keywords_path = os.path.join(current_keywords_folder, files[0])
            try:
                current_keywords_data = pd.read_csv(current_keywords_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    current_keywords_data = pd.read_csv(current_keywords_path, encoding='ISO-8859-1')
                except UnicodeDecodeError:
                    current_keywords_data = pd.read_csv(current_keywords_path, encoding='utf-16')
        else:
            st.error("No CSV files found in the current_keywords folder.")
    else:
        st.error("Current keywords folder not found.")
    
    return current_keywords_data

# Load potential keywords from any text file in the "potential_keywords" folder
def load_potential_keywords():
    potential_keywords_folder = os.path.join(os.getcwd(), "potential_keywords")

    # Check if the folder exists
    if os.path.exists(potential_keywords_folder):
        st.write(f"Found potential_keywords folder at: {potential_keywords_folder}")  # Debug statement
        # List all text files in the folder
        files = [f for f in os.listdir(potential_keywords_folder) if f.endswith('.txt')]
        if files:
            st.write(f"Text files found: {files}")  # Debug statement
            # Pick the first text file
            potential_keywords_path = os.path.join(potential_keywords_folder, files[0])
            return read_text_file(potential_keywords_path)
        else:
            st.error("No text files found in the potential_keywords folder.")
            return []
    else:
        st.error("Potential keywords folder not found.")
        return []

# Initialize OpenAI client using Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

# Extract context from knowledge files
knowledge_context = extract_context_from_knowledge()

# Load current keywords data
current_keywords_data = load_current_keyword_data()

# Load potential keywords
potential_keywords_list = load_potential_keywords()

# Sidebar
with st.sidebar:
    st.title('📊 Advanced Data Analysis Dashboard')
    
    uploaded_files = st.file_uploader(
        "Upload your qualitative data files (CSV, JSON, TXT)", 
        type=["csv", "json", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Load data with encoding handling
        def load_data(uploaded_files):
            dataframes = []
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.type == "application/json":
                        df = pd.read_json(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        df = pd.read_csv(uploaded_file, delimiter='\t')
                    else:
                        st.error("Unsupported file type!")
                        continue
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        try:
                            df = pd.read_csv(uploaded_file, encoding='utf-16')
                        except pd.errors.EmptyDataError:
                            st.error(f"The file {uploaded_file.name} is empty or has no columns to parse. Skipping this file.")
                            continue
                dataframes.append(df)
            return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        
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
                ["Topic Modeling", "Sentiment Analysis", "Word Cloud", "Topic Clustering", "Keyword Search Volume"]
            )

# Main Dashboard
if uploaded_files and data is not None and text_columns:
    # Preprocess text data
    data['processed_text'] = data[text_columns].astype(str).apply(lambda x: ' '.join(x), axis=1).apply(lambda x: x.lower())

    # Executive Summary
    st.markdown("## Executive Summary")
    st.markdown(f"- **Data Overview**: The dataset contains {len(data)} rows and the following selected columns: {', '.join(text_columns)}.")
    
    # Dashboard layout
    st.markdown("## Data Analysis Results")
    col1, col2 = st.columns([2, 3], gap="medium")
    
    with col1:
        # LLM Chat Interface
        st.markdown("### LLM Chat Interface")
        if text_columns:
            # Create a data summary for context based on selected columns
            column_descriptions = ""
            for col in text_columns:
                sample_values = ', '.join(data[col].astype(str).sample(3).values)
                column_descriptions += f"Column '{col}' has sample values like: {sample_values}. "

            # Include knowledge context in the prompt
            data_summary = f"The dataset contains {len(data)} rows and the following columns: {', '.join(text_columns)}. {column_descriptions} Knowledge context includes: {knowledge_context[:1000]}..."  # Limiting context to first 1000 characters
            
            # Display chat input and messages
            chat_container = st.container()
            with chat_container:
                if prompt := st.chat_input("Ask a question about the data"):
                    # Display user message
                    st.chat_message("user").write(prompt)
                    
                    # Query the LLM with data context
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": f"You are an expert data analyst.Use UK ENglish. Use the following dataset details and additional context to answer the user's questions: {data_summary}"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=4000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    response_text = response.choices[0].message.content
                    
                    # Display assistant response
                    st.chat_message("assistant").write(response_text)
        
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
        
        # Display Current and Potential Keyword Search Volume
        if "Keyword Search Volume" in analysis_options:
            st.markdown("### Current Keyword Search Volume")

            if current_keywords_data is not None:
                # Ensure all keywords are displayed, including those with empty search volumes
                if 'Keyword' in current_keywords_data.columns and 'Avg. monthly searches' in current_keywords_data.columns:
                    current_keywords_data['Search Volume'] = pd.to_numeric(current_keywords_data['Avg. monthly searches'], errors='coerce')
                    st.dataframe(current_keywords_data[['Keyword', 'Search Volume']].sort_values(by='Search Volume', ascending=False, na_position='last'))
                else:
                    st.error("Current keyword data does not have the required columns 'Keyword' and 'Avg. monthly searches'.")

            st.markdown("### Potential Keywords for Improvement")

            # Display potential keywords as a list
            if potential_keywords_list:
                st.write("Here are some potential keywords for improvement:")
                for keyword in potential_keywords_list:
                    st.write(f"- {keyword}")
            else:
                st.write("No potential keywords found or file is empty.")

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
            
            # Adjust perplexity to be less than the number of topics
            perplexity = min(30, len(lda.components_) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_results = tsne.fit_transform(lda.components_)
            cluster_df = pd.DataFrame(tsne_results, columns=['X', 'Y'])
            cluster_df['Cluster'] = topics_clustered
            
            # Scatter plot for clusters
            scatter_plot = px.scatter(cluster_df, x='X', y='Y', color='Cluster', color_discrete_sequence=selected_color_theme)
            scatter_plot.update_layout(template="plotly_dark")
            st.plotly_chart(scatter_plot, use_container_width=True)
            
            # LLM Interpretation for Each Cluster
            st.markdown("### LLM Interpretations of Clusters")
            interpretations = []
            for cluster_num in range(n_topics):
                cluster_text = " ".join(vectorizer.get_feature_names_out()[topics_df.iloc[cluster_num].argsort()[-10:]])
                interpretation_prompt = f"Interpret the following cluster of topics based on the context: {cluster_text}. Additional context: {knowledge_context[:1000]}..."  # Limiting context to first 1000 characters
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a data analyst. Use UK ENglish."},
                              {"role": "user", "content": interpretation_prompt}],
                    temperature=1,
                    max_tokens=1000,
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
            - **Dashboard Features**: Provides qualitative data analysis including topic modeling, sentiment analysis, clustering of topics, keyword search volume matching, and visualization tools like word clouds and scatter plots.
            - **Customization**: Users can choose the data columns for analysis and select different color themes for visualizations.
            - **Data Handling**: Supports multiple file uploads and handles different formats (CSV, JSON, TXT).
            - **Keyword Analysis**: Displays current keyword search volumes and potential keywords for improvement based on Google Keyword Planner data.
            - **Knowledge Integration**: Utilizes additional context from uploaded markdown and PDF files to enrich LLM responses and cluster interpretations.
        """)

else:
    if uploaded_files:
        st.error("Please select at least one text column for analysis.")
    else:
        st.info("Upload files to start analysis.")

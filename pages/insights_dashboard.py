import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from streamlit_tags import st_tags
from scrapegraphai import ScrapeGraphAI

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Access API keys from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]
scrapegraph_api_key = st.secrets["scrapegraph"]["api_key"]

# Initialize OpenAI client
openai_client = OpenAI(api_key=api_key)

# Initialize ScrapeGraphAI client
scrapegraph_client = ScrapeGraphAI(api_key=scrapegraph_api_key)

# Function to preprocess text data using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Function to chunk text into manageable sizes for the model
def chunk_text(text, chunk_size=2000):
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence.split()) > chunk_size:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence.split())
    if current_chunk:
        yield ' '.join(current_chunk)

# Function to extract text from multiple URLs and documents using ScrapeGraphAI
def extract_text_from_sources(sources):
    text_data = []
    try:
        results = scrapegraph_client.scrape_bulk(sources)  # Use scrape_bulk for URLs and documents
        for result in results:
            if result['success']:
                text_data.append(result['data']['text'])
            else:
                st.error(f"Error scraping the source: {result['error']}")
    except Exception as e:
        st.error(f"Exception occurred during scraping: {e}")
    return text_data

# Function to process uploaded files of various formats
def process_uploaded_files(files):
    file_paths = []
    for file in files:
        with open(file.name, 'wb') as f:
            f.write(file.getbuffer())
            file_paths.append(file.name)
    return file_paths

# Function to perform sentiment analysis
def perform_sentiment_analysis(texts):
    sentiments = [sia.polarity_scores(text) for text in texts]
    sentiment_df = pd.DataFrame(sentiments)
    return sentiment_df

# Function to perform K-means clustering
def perform_kmeans_clustering(texts, n_clusters=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    return clusters

# Function to generate advanced data graphs using Plotly
def generate_advanced_graph(data, graph_type="scatter"):
    if graph_type == "scatter":
        fig = px.scatter(data, x='x', y='y', color='label')
    elif graph_type == "line":
        fig = px.line(data, x='x', y='y', color='label')
    elif graph_type == "bar":
        fig = px.bar(data, x='x', y='y', color='label')
    st.plotly_chart(fig)

# Function to generate a comprehensive and relevant response using GPT-4
def generate_relevant_response(data, query):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant that provides concise and accurate answers to the user's questions based on the data provided."},
            {"role": "user", "content": f"Based on the following data: {data}. {query}"}
        ],
        temperature=0.3,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0.30,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Sidebar for data upload and configuration
st.sidebar.title("Data Upload and Configuration")

# Multi-line text input for URLs
urls_input = st.sidebar.text_area("Enter URLs to scrape data (one per line):")
urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

# Multiple file uploader
uploaded_files = st.sidebar.file_uploader("Upload multiple files", type=["xlsx", "csv", "pdf", "docx"], accept_multiple_files=True)

# Button to start processing
if st.sidebar.button("Submit"):
    all_texts = []
    document_metadata = []

    # Prepare sources for ScrapeGraphAI
    sources = urls

    # Process uploaded files and add to sources
    if uploaded_files:
        file_paths = process_uploaded_files(uploaded_files)
        sources.extend(file_paths)
        document_metadata.extend([{"source": file.name, "type": file.type} for file in uploaded_files])

    # Extract text from URLs and documents using ScrapeGraphAI
    if sources:
        text_data = extract_text_from_sources(sources)
        all_texts.extend(text_data)
        document_metadata.extend([{"source": source, "type": "url" if source in urls else "file"} for source in sources])

    # Store all collected texts and metadata in session state
    st.session_state['all_texts'] = all_texts
    st.session_state['document_metadata'] = document_metadata

    # Notify user that data has been scraped
    st.sidebar.success("Data has been successfully scraped.")

# Display data frames and allow filtering in the sidebar
if 'all_texts' in st.session_state and st.session_state['all_texts']:
    st.sidebar.header("Uploaded Data Preview and Filtering")

    # Placeholder for potential future data frame filtering

    # Option to query all text data using chatbot
    if st.sidebar.button("Query All Data"):
        if not st.session_state['all_texts']:
            st.sidebar.error("No data available to query. Please upload or scrape data first.")
        else:
            st.sidebar.success("All data is ready for querying.")

# Main Chatbot Interface
st.header("Chatbot for Data Analysis")

# Sticky tags feature and document selection dropdown
st.markdown(
    """
    <style>
    .sticky {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        background-color: white;
        padding: 10px;
        z-index: 1000;
        border-bottom: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sticky container for document selection and tags
st.markdown('<div class="sticky">', unsafe_allow_html=True)

# Document selection dropdown to select multiple documents
if 'document_metadata' in st.session_state:
    document_options = [f"{doc['source']} ({doc['type']})" for doc in st.session_state['document_metadata']]
    selected_documents = st.multiselect("Select documents to query:", document_options)
else:
    st.error("No documents available. Please upload data on the data page.")

# Define standard analysis chips
standard_chips = ["Sentiment Analysis", "K-means Clustering", "Advanced Graph"]
selected_chips = st_tags(
    label='Select Analysis Tools:',
    text='Press enter to add more',
    value=[],
    suggestions=standard_chips,
    maxtags=10,
    key='1'
)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input for chatbot
if prompt := st.chat_input("Ask a question about the selected documents:"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the query based on selected documents or all text data
    if 'all_texts' in st.session_state and selected_documents:
        selected_texts = [st.session_state['all_texts'][document_options.index(doc)] for doc in selected_documents]
        combined_text = ' '.join(selected_texts)

        # Query the data using GPT-4
        text_chunks = list(chunk_text(combined_text))
        responses = []
        for chunk in text_chunks:
            response = generate_relevant_response(chunk, prompt)
            responses.append(response)
        full_response = " ".join(responses)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("No data available. Please upload or select data for querying.")

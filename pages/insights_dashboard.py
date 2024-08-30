import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
from bs4 import BeautifulSoup, Comment
import time
import PyPDF2
import docx
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from streamlit_tags import st_tags

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Access API keys from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
openai_client = OpenAI(api_key=api_key)

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

# Function to extract text from URLs
def extract_text_from_urls(urls):
    text_data = []
    for url in urls:
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            texts = soup.find_all(string=True)
            visible_texts = filter(tag_visible, texts)
            text_data.append(" ".join(t.strip() for t in visible_texts))
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching the URL: {e}")
    return text_data

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    return '\n'.join(full_text)

# Function to process uploaded files of various formats
def process_uploaded_files(files):
    text_data = []
    data_frames = {}
    for file in files:
        if file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            df = pd.read_excel(file)
            text_data.extend(df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist())
            data_frames[file.name] = df
        elif file.type == 'text/csv':
            df = pd.read_csv(file)
            text_data.extend(df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist())
            data_frames[file.name] = df
        elif file.type == 'application/pdf':
            text_data.append(extract_text_from_pdf(file))
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text_data.append(extract_text_from_docx(file))
    return text_data, data_frames

# Function to check if a tag is visible
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

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

# Function to handle simple queries using a smaller model
def handle_simple_query(text, query):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that provides specific information from the text."},
            {"role": "user", "content": f"Based on the following text: {text}. {query}"}
        ],
        temperature=0.3,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
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
    data_frames = {}
    document_metadata = []

    # Extract text from URLs
    if urls:
        url_texts = extract_text_from_urls(urls)
        all_texts.extend(url_texts)
        document_metadata.extend([{"source": url, "type": "url"} for url in urls])

    # Process uploaded files
    if uploaded_files:
        file_texts, file_data_frames = process_uploaded_files(uploaded_files)
        all_texts.extend(file_texts)
        data_frames.update(file_data_frames)
        document_metadata.extend([{"source": file.name, "type": file.type} for file in uploaded_files])

    # Store all collected texts, data frames, and metadata in session state
    st.session_state['all_texts'] = all_texts
    st.session_state['data_frames'] = data_frames
    st.session_state['document_metadata'] = document_metadata

    # Notify user that data has been scraped
    st.sidebar.success("Data has been successfully scraped.")

# Display data frames and allow filtering in the sidebar
if 'data_frames' in st.session_state and st.session_state['data_frames']:
    st.sidebar.header("Uploaded Data Preview and Filtering")

    # Select a data frame to preview
    data_frame_names = list(st.session_state['data_frames'].keys())
    selected_data_frame_name = st.sidebar.selectbox("Select a data frame to preview and filter:", data_frame_names)

    # Display and filter the selected data frame
    selected_data_frame = st.session_state['data_frames'][selected_data_frame_name]
    st.sidebar.write(f"Preview of {selected_data_frame_name}:")

    # Show the dataframe and add filtering options
    st.sidebar.dataframe(selected_data_frame)

    # Create filtering options dynamically based on the columns
    filter_columns = st.sidebar.multiselect("Select columns to filter by", selected_data_frame.columns.tolist())
    
    filtered_data_frame = selected_data_frame

    for column in filter_columns:
        unique_values = filtered_data_frame[column].unique()
        selected_values = st.sidebar.multiselect(f"Filter {column}", unique_values)
        if selected_values:
            filtered_data_frame = filtered_data_frame[filtered_data_frame[column].isin(selected_values)]

    # Option to query filtered data using chatbot
    if st.sidebar.button("Query Filtered Data"):
        if filtered_data_frame.empty:
            st.sidebar.error("The filtered data frame is empty. Please adjust your filters.")
        else:
            # Convert filtered DataFrame to text for chatbot querying
            filtered_data_text = ' '.join(filtered_data_frame.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist())
            st.session_state['filtered_data_text'] = filtered_data_text
            st.sidebar.success("Filtered data is ready for querying.")

    # Display the filtered data frame in the main area
    st.write("Filtered Data:")
    st.dataframe(filtered_data_frame)

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

    # Process the query based on selected documents or filtered data
    if 'all_texts' in st.session_state and selected_documents:
        selected_texts = [st.session_state['all_texts'][document_options.index(doc)] for doc in selected_documents]
        combined_text = ' '.join(selected_texts)

        # Handle simple queries using a smaller model
        simple_response = handle_simple_query(combined_text, prompt)
        if simple_response:
            st.write(simple_response)
        else:
            # Process the query based on selected chips
            if "Sentiment Analysis" in selected_chips:
                st.write("Performing sentiment analysis on the data...")
                sentiments = perform_sentiment_analysis(selected_texts)
                st.write(sentiments)
            
            if "K-means Clustering" in selected_chips:
                st.write("Performing K-means clustering on the data...")
                clusters = perform_kmeans_clustering(selected_texts)
                st.write(clusters)
            
            if "Advanced Graph" in selected_chips:
                st.write("Generating advanced graph based on the data...")
                data = pd.DataFrame({
                    'x': range(10),
                    'y': range(10),
                    'label': ['A']*5 + ['B']*5
                })
                generate_advanced_graph(data, graph_type="scatter")
            
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
    elif 'filtered_data_text' in st.session_state:
        # Use the filtered data text for queries
        filtered_text = st.session_state['filtered_data_text']

        # Handle simple queries using a smaller model
        simple_response = handle_simple_query(filtered_text, prompt)
        if simple_response:
            st.write(simple_response)
        else:
            # Query the data using GPT-4
            text_chunks = list(chunk_text(filtered_text))
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
        st.error("No data available. Please upload data or filter a data frame first.")

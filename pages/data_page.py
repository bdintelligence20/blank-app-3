import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
import seaborn as sns
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
import numpy as np

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

# Connect to Milvus Lite
client = MilvusClient("./milvus_demo.db")

# Function to preprocess text data using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Function to chunk text into manageable sizes for the model
def chunk_text(text, chunk_size=2000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Function to get embeddings using OpenAI and store in Milvus Lite
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text.strip():  # Ensure the text is not empty
        raise ValueError("Input text for embedding is empty.")
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Function to store embeddings in Milvus Lite
def store_embeddings(texts):
    data = []
    for text in texts:
        preprocessed_text = preprocess_text(text)
        if preprocessed_text:
            embedding = get_embedding(preprocessed_text)
            data.append({
                "content": text[:65535],  # Truncate to fit VARCHAR max_length
                "embedding": embedding
            })

    if data:
        client.insert("text_embeddings", data)
        client.load("text_embeddings")  # Ensure data is loaded into memory

# Function to create collection if it doesn't exist
def create_collection():
    if "text_embeddings" not in client.list_collections():
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)  # Adjust dimension as per your embeddings
        ]
        schema = CollectionSchema(fields, description="Text embeddings collection")
        client.create_collection("text_embeddings", schema)

# Function to create index for efficient querying
def create_index():
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    client.create_index("text_embeddings", "embedding", index_params)

# Function to extract text from URLs
def extract_text_from_urls(urls):
    text_data = []
    for url in urls:
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            texts = soup.find_all(text=True)
            visible_texts = filter(tag_visible, texts)
            text_data.append(" ".join(t.strip() for t in visible_texts))
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching the URL: {e}")
    return text_data

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = []
    for page in pdf_reader.pages:
        full_text.append(page.extract_text())
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

# Function to generate insights using GPT-4 for general data
def generate_insights(text):
    insights = []
    for chunk in chunk_text(text):
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert analyst."},
                {"role": "user", "content": f"This data comes from a questionnaire sent to business leaders. The answers describe the problems we are solving for existing customers and the issues our offerings address. Based on this data, identify the top 5 problems for each division, keeping each problem to one sentence. Cluster the responses by commonalities and provide meaningful insights without focusing on punctuation or stop words: {chunk}"}
            ],
            temperature=1,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        insights.append(response.choices[0].message.content.strip())
    return insights

# Function to generate insights using GPT-4 specifically for web scraping
def generate_web_insights(text):
    insights = []
    for chunk in chunk_text(text):
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing website content."},
                {"role": "user", "content": f"Based on the scraped content, identify key themes and insights. Provide a summary of the main points: {chunk}"}
            ],
            temperature=1,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        insights.append(response.choices[0].message.content.strip())
    return insights

# Function to generate cluster labels using GPT-4
def generate_cluster_label(text):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert analyst in identifying themes in text data."},
            {"role": "user", "content": f"Analyze the following responses and suggest a common theme or label for them: {text}"}
        ],
        temperature=1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to generate a comprehensive list of keywords and key phrases
def generate_comprehensive_keywords(text):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in keyword analysis and SEO."},
            {"role": "user", "content": f"Based on the following text, generate a comprehensive list of relevant keywords and key phrases, including both short-tail and long-tail terms: {text}"}
        ],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip().split('\n')

# Function to search embeddings in Milvus Lite
def search_embeddings(query_text, top_k=5):
    preprocessed_query = preprocess_text(query_text)
    query_embedding = get_embedding(preprocessed_query)
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    try:
        results = client.search(
            collection_name="text_embeddings",
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content"]
        )
    except Exception as e:
        st.error(f"Failed to query collection: {e}")
        return []
    return results

# Function to summarize long text using GPT-4
def summarize_text(text):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": f"Summarize the following text in a concise manner: {text}"}
        ],
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Store data and allow querying through a chatbot interface
st.title("Interactive Chatbot for Data Analysis")

# Multi-line text input for URLs
urls_input = st.text_area("Enter URLs to scrape data (one per line):")
urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

# Multiple file uploader
uploaded_files = st.file_uploader("Upload multiple files", type=["xlsx", "csv", "pdf", "docx"], accept_multiple_files=True)

# Button to start processing
if st.button("Submit"):
    all_texts = []
    data_frames = {}

    # Extract text from URLs
    if urls:
        url_texts = extract_text_from_urls(urls)
        all_texts.extend(url_texts)

    # Process uploaded files
    if uploaded_files:
        file_texts, file_data_frames = process_uploaded_files(uploaded_files)
        all_texts.extend(file_texts)
        data_frames.update(file_data_frames)

    # Store all collected texts and data frames in session state
    st.session_state['all_texts'] = all_texts
    st.session_state['data_frames'] = data_frames

    # Notify user that data has been scraped
    st.success("Data has been successfully scraped.")

    # Create collection if it doesn't exist
    create_collection()

    # Store texts in Milvus
    store_embeddings(all_texts)

    # Create an index for the collection
    create_index()

    # Notify user that data has been stored and indexed
    st.success("Data has been stored and indexed in Milvus.")

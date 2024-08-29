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
    st.success("Data has been successfully scraped.")

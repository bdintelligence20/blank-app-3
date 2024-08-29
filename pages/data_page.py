import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import time
import PyPDF2
import docx
import matplotlib.pyplot as plt
import seaborn as sns
from pymilvus import MilvusClient
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

# Create collection if it doesn't exist
if "text_embeddings" not in client.list_collections():
    client.create_collection(
        collection_name="text_embeddings",
        dimension=1536  # Adjust this dimension as per your embeddings
    )

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

# Function to clean division names
def clean_division_names(df):
    df['Division'] = df['Division (TD, TT, TA, Impactful)'].str.strip().replace({
        'TD': 'Talent Development',
        'TT': 'Talent Technology',
        'TA': 'Talent Advisory',
        'Impactful': 'Impactful',
        'Marketing': 'Marketing',
        'Markting': 'Marketing',  # Correcting 'Markting' to 'Marketing'
        'Corporate': 'Corporate'
    })
    return df

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

# Function to check if a tag is visible
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

# Function to scrape a single webpage
def scrape_page(url):
    try:
        response = requests.get(url, verify=False)  # Disable SSL verification
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.find_all(text=True)
        visible_texts = filter(tag_visible, texts)
        return " ".join(t.strip() for t in visible_texts)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

# Function to scrape the main URL and all links in the navigation header
def scrape_website(url):
    main_content = scrape_page(url)
    scraped_data = {"Main": main_content}

    try:
        response = requests.get(url, verify=False)
        soup = BeautifulSoup(response.content, 'html.parser')
        nav_links = soup.find_all('a', href=True)

        # Extract unique links from the navigation
        links = {link['href'] for link in nav_links if link['href'].startswith('/')}

        for link in links:
            full_url = f"{url.rstrip('/')}/{link.lstrip('/')}"
            scraped_data[full_url] = scrape_page(full_url)
            time.sleep(2)  # Adding delay to prevent rate limiting or being blocked

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the URL: {e}")
    
    return scraped_data

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
        if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or file.type == 'application/vnd.ms-excel':
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

# Function to extract text from multiple URLs
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

# Function to extract keywords from text data
def extract_keywords(texts, n=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    keyword_counts = pd.DataFrame({'Keyword': keywords, 'Count': counts}).sort_values(by='Count', ascending=False)
    return keyword_counts.head(n)

# Function to store embeddings in Milvus Lite
def store_embeddings(texts):
    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings.append(response['data'][0]['embedding'])
    data = [{"id": i, "vector": embeddings[i]} for i in range(len(embeddings))]
    client.insert(
        collection_name="text_embeddings",
        data=data
    )

# Function to search embeddings in Milvus Lite
def search_embeddings(query_text, top_k=5):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query_text
    )
    query_embedding = response['data'][0]['embedding']
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = client.search(
        collection_name="text_embeddings",
        data=[query_embedding],
        anns_field="vector",
        params=search_params,
        limit=top_k,
        output_fields=["vector"]
    )
    return results

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

    # Store embeddings in Milvus
    store_embeddings(all_texts)

    # Notify user that data has been scraped and stored
    st.success("Data has been successfully scraped, stored, and embeddings have been stored in Milvus.")

# Chatbot interface for querying embeddings
if 'all_texts' in st.session_state:
    st.write("### Chat with the Data")
    user_query = st.text_input("Ask a question about the data or request a graph:")
    
    if st.button("Submit Query"):
        if user_query.lower().startswith("graph"):
            # Generate a graph based on keywords or data patterns
            st.write("Generating graph based on the data...")
            keywords = extract_keywords(st.session_state['all_texts'], n=10)
            fig, ax = plt.subplots()
            sns.barplot(x='Keyword', y='Count', data=keywords, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            # Query the data using GPT-4
            combined_text = ' '.join(st.session_state['all_texts'])
            text_chunks = list(chunk_text(combined_text))
            responses = []
            for chunk in text_chunks:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an intelligent assistant that provides concise and accurate answers to the user's questions based on the data provided."},
                        {"role": "user", "content": f"Based on the following data: {chunk}. {user_query}"}
                    ],
                    temperature=0.3,  # Lower temperature for more concise responses
                    max_tokens=500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                responses.append(response.choices[0].message.content.strip())
            full_response = " ".join(responses)
            st.write(full_response)

        # Embedding search query
        search_results = search_embeddings(user_query)
        st.write("Search results for embeddings:")
        st.write(search_results)

# End of script


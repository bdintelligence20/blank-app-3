import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Access API keys from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to preprocess text data using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

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
    for file in files:
        if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or file.type == 'application/vnd.ms-excel':
            df = pd.read_excel(file)
            text_data.extend(df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist())
        elif file.type == 'text/csv':
            df = pd.read_csv(file)
            text_data.extend(df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist())
        elif file.type == 'application/pdf':
            text_data.append(extract_text_from_pdf(file))
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text_data.append(extract_text_from_docx(file))
    return text_data

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

# Function to generate insights using GPT-4
def generate_insights(text, content_type="general"):
    role_description = "You are an expert analyst." if content_type == "general" else "You are an expert in analyzing website content."
    task_description = (
        "This data comes from a questionnaire sent to business leaders. The answers describe the problems we are solving for existing customers and the issues our offerings address. "
        "Based on this data, identify the top 5 problems for each division, keeping each problem to one sentence. Cluster the responses by commonalities and provide meaningful insights without focusing on punctuation or stop words."
        if content_type == "general" else
        "Based on the scraped content, identify key themes and insights. Provide a summary of the main points."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": f"{task_description}: {text}"}
        ],
        temperature=1,
        max_tokens=16383,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to extract keywords from text data
def extract_keywords(texts, n=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    keyword_counts = pd.DataFrame({'Keyword': keywords, 'Count': counts}).sort_values(by='Count', ascending=False)
    return keyword_counts.head(n)

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
    all_insights = []

    # Extract text from URLs and generate insights
    if urls:
        url_texts = extract_text_from_urls(urls)
        all_texts.extend(url_texts)
        for text in url_texts:
            insights = generate_insights(text, content_type="web")
            all_insights.append(insights)

    # Process uploaded files and generate insights
    if uploaded_files:
        file_texts = process_uploaded_files(uploaded_files)
        all_texts.extend(file_texts)
        for text in file_texts:
            insights = generate_insights(text, content_type="general")
            all_insights.append(insights)

    # Store all collected texts and insights in session state
    st.session_state['all_texts'] = all_texts
    st.session_state['all_insights'] = all_insights
    st.success("Data has been scraped and analyzed successfully. You can now query the data.")

# Chatbot interface
if 'all_insights' in st.session_state and st.session_state['all_insights']:
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
            # Dynamically query the stored insights
            relevant_insights = []
            for insight in st.session_state['all_insights']:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an intelligent assistant that helps analyze data and answer questions."},
                        {"role": "user", "content": f"Based on the following insight: {insight}. {user_query}"}
                    ],
                    temperature=0.5,
                    max_tokens=16383,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                relevant_insights.append(response.choices[0].message.content.strip())
            
            # Display a single or concatenated response
            st.write("\n\n".join(relevant_insights))
else:
    st.info("Please submit data to scrape and process before querying.")

# Web Search Functionality
st.write("### Web Search")
search_query = st.text_input("Enter a search query for additional information:")
if st.button("Search Web"):
    # Implement a web search using an external API or tool (not implemented here)
    st.write("Web search feature is not implemented in this script.")

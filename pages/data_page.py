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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": f"This data comes from a questionnaire sent to business leaders. The answers describe the problems we are solving for existing customers and the issues our offerings address. Based on this data, identify the top 5 problems for each division, keeping each problem to one sentence. Cluster the responses by commonalities and provide meaningful insights without focusing on punctuation or stop words: {text}"}
        ],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to generate insights using GPT-4 specifically for web scraping
def generate_web_insights(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing website content."},
            {"role": "user", "content": f"Based on the scraped content, identify key themes and insights. Provide a summary of the main points: {text}"}
        ],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to generate cluster labels using GPT-4
def generate_cluster_label(text):
    response = client.chat.completions.create(
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
    response = client.chat.completions.create(
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

# Function to extract keywords from text data
def extract_keywords(texts, n=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    keyword_counts = pd.DataFrame({'Keyword': keywords, 'Count': counts}).sort_values(by='Count', ascending=False)
    return keyword_counts.head(n)

# Streamlit UI
st.title("Text Analysis with GPT-4")

# Multi-line text input for URLs
urls_input = st.text_area("Enter URLs to scrape data (one per line):")
urls = [url.strip() for url in urls_input.splitlines() if url.strip()]

# Multiple file uploader
uploaded_files = st.file_uploader("Upload multiple files", type=["xlsx", "csv", "pdf", "docx"], accept_multiple_files=True)

# Button to start processing
if st.button("Submit"):
    all_texts = []

    # Extract text from URLs
    if urls:
        url_texts = extract_text_from_urls(url)
        st.write("## Scraped Content from URLs")
        for i, text in enumerate(url_texts):
            st.write(f"### URL {i + 1}")
            st.write(text)
            all_texts.extend(url_texts)

    # Process uploaded files
    if uploaded_files:
        file_texts = process_uploaded_files(uploaded_files)
        st.write("## Content Extracted from Uploaded Files")
        for i, text in enumerate(file_texts):
            st.write(f"### File {i + 1}")
            st.write(text)
            all_texts.extend(file_texts)

    # Ensure we have text data to analyze
    if all_texts:
        # Extract keywords from all collected texts
        keyword_counts = extract_keywords(all_texts, n=20)
        st.write("### Short-Tail and Long-Tail Keywords for All Data")
        st.write(keyword_counts)

        # Save keywords to session state
        st.session_state['all_short_tail_keywords'] = keyword_counts[keyword_counts['Keyword'].str.split().str.len() == 1]
        st.session_state['all_long_tail_keywords'] = keyword_counts[keyword_counts['Keyword'].str.split().str.len() > 1]

        # Generate comprehensive list of keywords and key phrases
        all_keywords = st.session_state['all_short_tail_keywords']['Keyword'].tolist() + \
                       st.session_state['all_long_tail_keywords']['Keyword'].tolist()

        # Add any additional key phrases related to all text summaries
        combined_text = ' '.join(all_texts)
        additional_keywords = generate_comprehensive_keywords(combined_text)
        all_keywords.extend(additional_keywords)

        # Remove duplicates and clean the keyword list
        all_keywords = list(set(all_keywords))

        # Display the comprehensive list of keywords and key phrases
        st.write("### Comprehensive List of Keywords and Key Phrases")
        st.write(all_keywords)

        # Analyze and display insights using GPT-4
        st.write("### Insights from All Data")
        insights = generate_insights(combined_text)
        st.write(insights)

# Load and display previously stored data from session state
else:
    # Check if text data is in session state
    if 'all_short_tail_keywords' in st.session_state:
        st.write("### Short-Tail Keywords for All Data")
        st.write(st.session_state['all_short_tail_keywords'])
        
    if 'all_long_tail_keywords' in st.session_state:
        st.write("### Long-Tail Keywords for All Data")
        st.write(st.session_state['all_long_tail_keywords'])
    
    if 'excel_insights' in st.session_state:
        st.write("### Insights from All Data")
        st.write(st.session_state['excel_insights'])

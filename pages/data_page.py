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

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to preprocess text data using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

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

# Access the API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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

# URL input for scraping
url = st.text_input("Enter a URL to scrape data:")
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Display all results without any filtering
if st.button("Submit"):
    web_texts = []
    excel_texts = []

    # Check if URL is provided for scraping
    if url:
        scraped_data = scrape_website(url)
        if scraped_data:
            for page, text in scraped_data.items():
                st.write(f"### Scraped Content from {page}")
                st.write(text)
                web_texts.append(text)
                
                # Use LLM to generate insights from scraped text
                web_insights = generate_web_insights(text)
                st.write(f"#### Insights for {page}")
                st.write(web_insights)

            # Extract keywords for web scraped data
            if web_texts:
                keyword_counts = extract_keywords(web_texts, n=20)
                st.write("### Short-Tail and Long-Tail Keywords for Web Scraped Data")
                st.write(keyword_counts)

                # Save keywords to session state
                st.session_state['web_short_tail_keywords'] = keyword_counts[keyword_counts['Keyword'].str.split().str.len() == 1]
                st.session_state['web_long_tail_keywords'] = keyword_counts[keyword_counts['Keyword'].str.split().str.len() > 1]

            # Save web scraped data and insights to session state
            st.session_state['web_texts'] = web_texts
            st.session_state['web_insights'] = [generate_web_insights(text) for text in web_texts]

    # Check if an Excel file is uploaded
    if uploaded_file is not None:
        # Load Excel data
        @st.cache_data
        def load_data(file):
            data = pd.read_excel(file)
            data.fillna("", inplace=True)  # Fill NaN values with empty strings
            return data

        data = load_data(uploaded_file)

        # Clean division names
        data = clean_division_names(data)

        # Process Excel data without filtering
        st.write("## Raw Data from Excel")
        st.write(data)

        # Combine all relevant columns into one
        data['All_Problems'] = data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        data['Processed_Text'] = data['All_Problems'].apply(preprocess_text)
        excel_texts = data['Processed_Text'].tolist()

        # Perform text vectorization using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data['Processed_Text'])

        # Perform KMeans clustering
        num_clusters = st.slider('Select number of clusters:', 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(X)

        data['Cluster'] = kmeans.labels_

        # Initialize a list to store cluster labels
        cluster_labels = []

        # Generate labels for each cluster
        for cluster_num in range(num_clusters):
            cluster_data = data[data['Cluster'] == cluster_num]['All_Problems'].tolist()
            # Check if there is any data for this cluster
            if cluster_data:
                cluster_label = generate_cluster_label(' '.join(cluster_data))
                cluster_labels.append(cluster_label)
            else:
                # If no data is found for a cluster, append a placeholder
                cluster_labels.append("No data available for this cluster")

        # Ensure the number of cluster labels matches the number of clusters
        if len(cluster_labels) != num_clusters:
            st.error("Mismatch in number of clusters and cluster labels. Adjusting...")
            cluster_labels = cluster_labels[:num_clusters]

        # Save Excel data and insights to session state
        st.session_state['filtered_data'] = data
        st.session_state['cluster_labels'] = cluster_labels
        st.session_state['excel_texts'] = excel_texts
        st.session_state['excel_clusters'] = num_clusters
        st.session_state['excel_insights'] = [
            generate_insights(' '.join(data[data['Cluster'] == cluster_num]['All_Problems'].tolist())) 
            if len(data[data['Cluster'] == cluster_num]) > 0 else "No data available"
            for cluster_num in range(num_clusters)
        ]

        # Display the processed data and insights without filtering
        st.write("## Processed Data for Excel")
        st.write(data)

        for cluster_num in range(num_clusters):
            st.write(f"### Cluster {cluster_num + 1}: {cluster_labels[cluster_num]}")
            cluster_data = data[data['Cluster'] == cluster_num]['All_Problems'].tolist()
            if cluster_data:
                insights = generate_insights(' '.join(cluster_data))
                st.write(insights)
            else:
                st.write("No data available for this cluster.")

        # Extract keywords for Excel data
        if excel_texts:
            keyword_counts = extract_keywords(excel_texts, n=20)
            st.write("### Short-Tail and Long-Tail Keywords for Excel Data")
            st.write(keyword_counts)

            # Save keywords to session state
            st.session_state['excel_short_tail_keywords'] = keyword_counts[keyword_counts['Keyword'].str.split().str.len() == 1]
            st.session_state['excel_long_tail_keywords'] = keyword_counts[keyword_counts['Keyword'].str.split().str.len() > 1]

# Load and display previously stored data from session state
else:
    # Check if web data is in session state
    if 'web_texts' in st.session_state:
        st.write("## Web Data from Session")
        web_texts = st.session_state['web_texts']
        web_insights = st.session_state['web_insights']
        
        for i, text in enumerate(web_texts):
            st.write(f"### Scraped Content {i + 1}")
            st.write(text)
            st.write(f"#### Insights for Scraped Content {i + 1}")
            st.write(web_insights[i])

        # Display previously stored keywords
        if 'web_short_tail_keywords' in st.session_state and 'web_long_tail_keywords' in st.session_state:
            st.write("### Short-Tail Keywords for Web Data")
            st.write(st.session_state['web_short_tail_keywords'])
            st.write("### Long-Tail Keywords for Web Data")
            st.write(st.session_state['web_long_tail_keywords'])

    # Check if Excel data is in session state
    if 'filtered_data' in st.session_state:
        st.write("## Excel Data from Session")
        data = st.session_state['filtered_data']
        cluster_labels = st.session_state['cluster_labels']
        excel_insights = st.session_state['excel_insights']
        num_clusters = st.session_state['excel_clusters']

        st.write("## Processed Data for Excel")
        st.write(data)

        for cluster_num in range(num_clusters):
            st.write(f"### Cluster {cluster_num + 1}: {cluster_labels[cluster_num]}")
            cluster_data = data[data['Cluster'] == cluster_num]['All_Problems'].tolist()
            st.write(excel_insights[cluster_num])

        # Display previously stored keywords
        if 'excel_short_tail_keywords' in st.session_state and 'excel_long_tail_keywords' in st.session_state:
            st.write("### Short-Tail Keywords for Excel Data")
            st.write(st.session_state['excel_short_tail_keywords'])
            st.write("### Long-Tail Keywords for Excel Data")
            st.write(st.session_state['excel_long_tail_keywords'])

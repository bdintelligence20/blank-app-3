import os
import time
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

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

# Function to perform sentiment analysis on text
def analyze_sentiment(text):
    return sia.polarity_scores(text)

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
    return keyword_counts

# Streamlit UI
st.title("Text Analysis with GPT-4")

# URL input for scraping
url = st.text_input("Enter a URL to scrape data:")
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Submit button
if st.button("Submit"):
    all_texts = []

    # Check if URL is provided for scraping
    if url:
        scraped_data = scrape_website(url)
        if scraped_data:
            for page, text in scraped_data.items():
                st.write(f"### Scraped Content from {page}")
                st.write(text)
                
                # Use LLM to generate insights from scraped text
                web_insights = generate_web_insights(text)
                st.write(f"#### Insights for {page}")
                st.write(web_insights)

                # Extract keywords from scraped text
                keyword_counts = extract_keywords([text])
                short_tail_keywords = keyword_counts[keyword_counts['Keyword'].apply(lambda x: len(x.split()) == 1)]
                long_tail_keywords = keyword_counts[keyword_counts['Keyword'].apply(lambda x: len(x.split()) > 1)]

                st.write("### Short-Tail Keywords")
                st.write(short_tail_keywords)

                st.write("### Long-Tail Keywords")
                st.write(long_tail_keywords)

                # Append processed text to all_texts for further analysis
                all_texts.append(text)

    # Check if an Excel file is uploaded
    if uploaded_file is not None:
        # Load Excel data
        @st.cache_data
        def load_data(file):
            data = pd.read_excel(file)
            data.fillna("", inplace=True)  # Fill NaN values with empty strings
            return data

        data = load_data(uploaded_file)

        # Display division options and filter data
        division_options = data['Division (TD, TT, TA, Impactful)'].unique()
        selected_division = st.selectbox("Select a Division:", division_options)
        filtered_data = data[data['Division (TD, TT, TA, Impactful)'] == selected_division]

        # Combine all relevant columns into one
        filtered_data.loc[:, 'All_Problems'] = filtered_data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        filtered_data.loc[:, 'Processed_Text'] = filtered_data['All_Problems'].apply(preprocess_text)
        all_texts.extend(filtered_data['Processed_Text'].tolist())

        # Perform text vectorization using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(filtered_data['Processed_Text'])

        # Perform KMeans clustering
        num_clusters = st.slider('Select number of clusters:', 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(X)

        filtered_data.loc[:, 'Cluster'] = kmeans.labels_

        # Initialize a list to store cluster labels
        cluster_labels = []

        # Generate labels for each cluster
        for cluster_num in range(num_clusters):
            cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['All_Problems'].tolist()
            cluster_label = generate_cluster_label(' '.join(cluster_data))
            cluster_labels.append(cluster_label)

        # Display clusters, insights, and visualizations
        for cluster_num in range(num_clusters):
            st.write(f"### Cluster {cluster_num + 1}: {cluster_labels[cluster_num]}")
            cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['All_Problems'].tolist()
            insights = generate_insights(' '.join(cluster_data))
            st.write(insights)

            # Keyword Counts in Each Cluster
            cluster_data_processed = filtered_data[filtered_data['Cluster'] == cluster_num]['Processed_Text'].tolist()
            keyword_counts = extract_keywords(cluster_data_processed)

            # Short-Tail and Long-Tail Keywords for Each Cluster
            short_tail_keywords = keyword_counts[keyword_counts['Keyword'].apply(lambda x: len(x.split()) == 1)]
            long_tail_keywords = keyword_counts[keyword_counts['Keyword'].apply(lambda x: len(x.split()) > 1)]

            st.write("### Short-Tail Keywords for Cluster")
            st.write(short_tail_keywords)

            st.write("### Long-Tail Keywords for Cluster")
            st.write(long_tail_keywords)

            # Plotting the keyword counts
            fig, ax = plt.subplots()
            sns.barplot(x='Count', y='Keyword', data=keyword_counts.head(10), ax=ax)
            ax.set_xlabel('Count')
            ax.set_ylabel('Keyword')
            ax.set_title(f'Keyword Counts for Cluster {cluster_num + 1}')
            st.pyplot(fig)

            # Sentiment Analysis for Each Cluster
            sentiments = [analyze_sentiment(text) for text in cluster_data]
            sentiment_df = pd.DataFrame(sentiments)
            st.write(f"Sentiment Analysis for Cluster {cluster_num + 1}")
            st.write(sentiment_df.describe())

        # Division-Specific Problem Frequency
        st.write("### Division-Specific Problem Frequency")
        problem_freq = filtered_data['Division (TD, TT, TA, Impactful)'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=problem_freq.index, y=problem_freq.values, ax=ax)
        ax.set_xlabel('Division')
        ax.set_ylabel('Frequency')
        ax.set_title('Problem Frequency by Division')
        st.pyplot(fig)

        # Cluster Distribution
        st.write("### Cluster Distribution")
        cluster_counts = filtered_data['Cluster'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Responses')
        ax.set_title('Distribution of Responses Across Clusters')
        st.pyplot(fig)

        # Inter-Cluster Similarity
        st.write("### Inter-Cluster Similarity")
        similarity_matrix = cosine_similarity(X)
        fig, ax = plt.subplots()
        sns.heatmap(similarity_matrix, cmap='viridis', ax=ax)
        ax.set_title('Inter-Cluster Similarity')
        st.pyplot(fig)

        # Display the processed data
        st.write(filtered_data)


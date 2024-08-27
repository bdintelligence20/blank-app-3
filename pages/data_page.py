import os
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
import re

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to preprocess text data using spaCy
def preprocess_text(text):
    # Remove emails, URLs, and personal information
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\b[A-Z][a-z]*\b', '', text)  # Remove proper nouns (assumed to be names)
    
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Function to clean and standardize division names
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

# Function to generate insights using GPT-4
def generate_insights(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert business analyst focusing on organizational challenges and solutions."},
            {"role": "user", "content": f"This data is derived from a questionnaire where business leaders describe challenges and solutions. Focus on identifying the key business problems and challenges without considering any contact information, names, or geographic locations. Provide the top 5 problems for each division based on the text data: {text}"}
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
            {"role": "system", "content": "You are an expert business analyst specializing in identifying themes in organizational challenges."},
            {"role": "user", "content": f"Analyze the following responses and suggest a common theme or label for them. Focus on the business problems described and ignore any mention of contact information or geographical locations: {text}"}
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

# Streamlit UI
st.title("Division-Specific Text Analysis with GPT-4")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load Excel data
    @st.cache_data
    def load_data(file):
        data = pd.read_excel(file)
        data.fillna("", inplace=True)  # Fill NaN values with empty strings
        data = clean_division_names(data)  # Clean and standardize division names
        return data

    data = load_data(uploaded_file)

    # Display division options and filter data
    division_options = data['Division'].unique()
    selected_division = st.selectbox("Select a Division:", division_options)
    filtered_data = data[data['Division'] == selected_division]

    # Combine all relevant columns into one
    filtered_data['All_Problems'] = filtered_data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    filtered_data['Processed_Text'] = filtered_data['All_Problems'].apply(preprocess_text)

    # Perform text vectorization using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(filtered_data['Processed_Text'])

    # Perform KMeans clustering specific to the selected division
    num_clusters = st.slider('Select number of clusters:', 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)

    filtered_data['Cluster'] = kmeans.labels_

    # Initialize a list to store cluster labels
    cluster_labels = []

    # Generate labels for each cluster
    for cluster_num in range(num_clusters):
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['All_Problems'].tolist()
        cluster_label = generate_cluster_label(' '.join(cluster_data))
        cluster_labels.append(cluster_label)

    # Display division summary
    st.write(f"## Summary of {selected_division} Division's Submitted Problems")
    division_summary = ' '.join(filtered_data['All_Problems'].tolist())
    st.write(division_summary)

    # Generate and display insights
    st.write(f"## Insights for {selected_division} Division")
    division_insights = generate_insights(division_summary)
    st.write(division_insights)

    # Display clusters, insights, and visualizations
    for cluster_num in range(num_clusters):
        st.write(f"### Cluster {cluster_num + 1}: {cluster_labels[cluster_num]}")
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['All_Problems'].tolist()
        insights = generate_insights(' '.join(cluster_data))
        st.write(insights)

        # Keyword Counts in Each Cluster
        cluster_data_processed = filtered_data[filtered_data['Cluster'] == cluster_num]['Processed_Text'].tolist()
        count_vectorizer = CountVectorizer(stop_words='english', max_features=10)
        X_cluster = count_vectorizer.fit_transform(cluster_data_processed)
        keywords = count_vectorizer.get_feature_names_out()
        counts = X_cluster.toarray().sum(axis=0)
        keyword_counts = pd.DataFrame({'Keyword': keywords, 'Count': counts}).sort_values(by='Count', ascending=False)

        # Plotting the keyword counts
        fig, ax = plt.subplots()
        sns.barplot(x='Count', y='Keyword', data=keyword_counts, ax=ax)
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
    problem_freq = filtered_data['Division'].value_counts()
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

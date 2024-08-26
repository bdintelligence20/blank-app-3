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

# Function to generate insights using GPT-4
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

# Streamlit UI
st.title("Text Analysis with GPT-4")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

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

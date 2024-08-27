import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize NLTK's SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis on text
def analyze_sentiment(text):
    return sia.polarity_scores(text)

# Function to extract keywords from text data
def extract_keywords(texts, n=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    keyword_counts = pd.DataFrame({'Keyword': keywords, 'Count': counts}).sort_values(by='Count', ascending=False)
    return keyword_counts.head(n)

# Page title
st.title("Insights Dashboard")

# Ensure data is available
if 'filtered_data' not in st.session_state or 'cluster_labels' not in st.session_state:
    st.write("No data available. Please go to the 'Data Page' and submit data first.")
else:
    # Load data from session state
    filtered_data = st.session_state['filtered_data']
    cluster_labels = st.session_state['cluster_labels']
    excel_texts = st.session_state['excel_texts']
    excel_clusters = st.session_state['excel_clusters']
    excel_insights = st.session_state.get('excel_insights', [])
    web_texts = st.session_state.get('web_texts', [])

    # Display the top 5 problems for each division
    st.write("## Top 5 Problems per Division")
    for division in filtered_data['Division'].unique():
        division_data = filtered_data[filtered_data['Division'] == division]
        problems = division_data['All_Problems'].tolist()
        st.write(f"### {division}")
        st.write(problems[:5])

    # Display cluster distribution
    st.write("## Cluster Distribution for Excel Data")
    cluster_counts = filtered_data['Cluster'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Responses')
    ax.set_title('Distribution of Responses Across Clusters')
    st.pyplot(fig)

    # Division-specific problem frequency
    st.write("## Division-Specific Problem Frequency")
    problem_freq = filtered_data['Division'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=problem_freq.index, y=problem_freq.values, ax=ax)
    ax.set_xlabel('Division')
    ax.set_ylabel('Frequency')
    ax.set_title('Problem Frequency by Division')
    st.pyplot(fig)

    # Sentiment analysis for each cluster
    st.write("## Sentiment Analysis for Each Cluster in Excel Data")
    sentiments = [analyze_sentiment(text) for text in excel_texts]
    sentiment_df = pd.DataFrame(sentiments)
    st.write(sentiment_df.describe())

    # Inter-cluster similarity for Excel data
    st.write("## Inter-Cluster Similarity for Excel Data")
    similarity_matrix = cosine_similarity(filtered_data['Processed_Text'])
    fig, ax = plt.subplots()
    sns.heatmap(similarity_matrix, cmap='viridis', ax=ax)
    ax.set_title('Inter-Cluster Similarity')
    st.pyplot(fig)

    # Combined keyword analysis from web and Excel data
    if web_texts:
        all_texts = web_texts + excel_texts
        st.write("## Combined Keyword Analysis from Web and Excel Data")
        keyword_counts = extract_keywords(all_texts, n=20)
        short_tail_keywords = keyword_counts[keyword_counts['Keyword'].str.split().str.len() == 1]
        long_tail_keywords = keyword_counts[keyword_counts['Keyword'].str.split().str.len() > 1]

        st.write("### Combined Short-Tail Keywords")
        st.write(short_tail_keywords)

        st.write("### Combined Long-Tail Keywords")
        st.write(long_tail_keywords)

        fig, ax = plt.subplots()
        sns.barplot(x='Count', y='Keyword', data=keyword_counts, ax=ax)
        ax.set_xlabel('Count')
        ax.set_ylabel('Keyword')
        ax.set_title('Combined Keyword Counts from Web and Excel Data')
        st.pyplot(fig)

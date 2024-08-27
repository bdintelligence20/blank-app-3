import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Ensure Streamlit session state is initialized for expected variables
if 'cluster_data' not in st.session_state or 'cluster_labels' not in st.session_state:
    st.warning("Please process the data on the Data Page before viewing the Insights Dashboard.")
else:
    st.title("Insights Dashboard")

    # Retrieve processed data from session state
    filtered_data = st.session_state['cluster_data']
    cluster_labels = st.session_state['cluster_labels']
    excel_texts = st.session_state['excel_texts']
    web_texts = st.session_state['web_texts']

    # Visualize Cluster Labels and Insights
    for cluster_num, cluster_label in enumerate(cluster_labels):
        st.write(f"### Cluster {cluster_num + 1}: {cluster_label}")
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['All_Problems'].tolist()
        st.write(f"#### Problems for Cluster {cluster_num + 1}")
        st.write(cluster_data)

    # Keyword Extraction and Visualization
    def plot_keyword_counts(keyword_counts, title):
        fig, ax = plt.subplots()
        sns.barplot(x='Count', y='Keyword', data=keyword_counts, ax=ax)
        ax.set_xlabel('Count')
        ax.set_ylabel('Keyword')
        ax.set_title(title)
        st.pyplot(fig)

    # Separate Short-Tail and Long-Tail Keywords
    def separate_keywords(keyword_counts):
        short_tail = keyword_counts[keyword_counts['Keyword'].str.split().str.len() == 1]
        long_tail = keyword_counts[keyword_counts['Keyword'].str.split().str.len() > 1]
        return short_tail, long_tail

    # Excel Data Keyword Analysis
    st.write("### Keyword Analysis for Excel Data")
    if excel_texts:
        excel_keyword_counts = extract_keywords(excel_texts, n=20)
        short_tail, long_tail = separate_keywords(excel_keyword_counts)

        st.write("#### Short-Tail Keywords for Excel Data")
        st.write(short_tail)

        st.write("#### Long-Tail Keywords for Excel Data")
        st.write(long_tail)

        plot_keyword_counts(excel_keyword_counts, 'Keyword Counts for Excel Data')

    # Web Scraped Data Keyword Analysis
    st.write("### Keyword Analysis for Web Scraped Data")
    if web_texts:
        web_keyword_counts = extract_keywords(web_texts, n=20)
        short_tail, long_tail = separate_keywords(web_keyword_counts)

        st.write("#### Short-Tail Keywords for Web Scraped Data")
        st.write(short_tail)

        st.write("#### Long-Tail Keywords for Web Scraped Data")
        st.write(long_tail)

        plot_keyword_counts(web_keyword_counts, 'Keyword Counts for Web Scraped Data')

    # Combined Keyword Analysis
    st.write("### Combined Keyword Analysis from Web and Excel Data")
    all_texts = web_texts + excel_texts
    if all_texts:
        combined_keyword_counts = extract_keywords(all_texts, n=20)
        short_tail, long_tail = separate_keywords(combined_keyword_counts)

        st.write("#### Combined Short-Tail Keywords")
        st.write(short_tail)

        st.write("#### Combined Long-Tail Keywords")
        st.write(long_tail)

        plot_keyword_counts(combined_keyword_counts, 'Combined Keyword Counts from Web and Excel Data')

    # Division-Specific Problem Frequency for Excel data
    st.write("### Division-Specific Problem Frequency")
    problem_freq = filtered_data['Division'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=problem_freq.index, y=problem_freq.values, ax=ax)
    ax.set_xlabel('Division')
    ax.set_ylabel('Frequency')
    ax.set_title('Problem Frequency by Division')
    st.pyplot(fig)

    # Cluster Distribution for Excel data
    st.write("### Cluster Distribution")
    cluster_counts = filtered_data['Cluster'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Responses')
    ax.set_title('Distribution of Responses Across Clusters')
    st.pyplot(fig)

    # Inter-Cluster Similarity for Excel data
    st.write("### Inter-Cluster Similarity")
    similarity_matrix = cosine_similarity(filtered_data['Processed_Text'].tolist())
    fig, ax = plt.subplots()
    sns.heatmap(similarity_matrix, cmap='viridis', ax=ax)
    ax.set_title('Inter-Cluster Similarity')
    st.pyplot(fig)

    # Sentiment Analysis for Each Cluster in Excel data
    for cluster_num in range(len(cluster_labels)):
        st.write(f"### Sentiment Analysis for Cluster {cluster_num + 1}")
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['Processed_Text'].tolist()
        sentiments = [analyze_sentiment(text) for text in cluster_data]
        sentiment_df = pd.DataFrame(sentiments)
        st.write(sentiment_df.describe())

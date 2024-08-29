import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymilvus import MilvusClient
import nltk
from streamlit_tags import st_tags, st_tags_sidebar

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

# Function to extract keywords from text data
def extract_keywords(texts, n=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    keyword_counts = pd.DataFrame({'Keyword': keywords, 'Count': counts}).sort_values(by='Count', ascending=False)
    return keyword_counts.head(n)

# Function to get embeddings using OpenAI and store in Milvus Lite
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text.strip():  # Ensure the text is not empty
        raise ValueError("Input text for embedding is empty.")
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Function to search embeddings in Milvus Lite
def search_embeddings(query_text, top_k=5):
    # Preprocess query text
    preprocessed_query = preprocess_text(query_text)
    query_embedding = get_embedding(preprocessed_query)
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    try:
        results = client.search(
            collection_name="text_embeddings",
            data=[query_embedding],
            anns_field="vector",
            params=search_params,
            limit=top_k,
            output_fields=["vector"]
        )
    except Exception as e:
        st.error(f"Failed to query collection: {e}")
        return []
    return results

# Function to summarize long text using GPT-4
def summarize_text(text):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": f"Summarize the following text in a concise manner: {text}"}
        ],
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to generate a comprehensive and relevant response using GPT-4
def generate_relevant_response(data, query):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant that provides concise and accurate answers to the user's questions based on the data provided."},
            {"role": "user", "content": f"Based on the following data: {data}. {query}"}
        ],
        temperature=0.3,  # Lower temperature for more concise responses
        max_tokens=4000,  # Allow more tokens for comprehensive answers
        top_p=1,
        frequency_penalty=0.30,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to generate a pie chart based on keyword counts
def generate_pie_chart(data):
    fig, ax = plt.subplots()
    ax.pie(data['Count'], labels=data['Keyword'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Function to perform sentiment analysis
def perform_sentiment_analysis(texts):
    sentiments = [sia.polarity_scores(text) for text in texts]
    sentiment_df = pd.DataFrame(sentiments)
    return sentiment_df

# Function to perform K-means clustering
def perform_kmeans_clustering(texts, n_clusters=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    return clusters

# Function to generate advanced data graphs using Plotly
def generate_advanced_graph(data, graph_type="scatter"):
    if graph_type == "scatter":
        fig = px.scatter(data, x='x', y='y', color='label')
    elif graph_type == "line":
        fig = px.line(data, x='x', y='y', color='label')
    elif graph_type == "bar":
        fig = px.bar(data, x='x', y='y', color='label')
    st.plotly_chart(fig)

# Store data and allow querying through a chatbot interface
st.title("Interactive Chatbot for Data Analysis")

# Define standard analysis chips
standard_chips = ["Sentiment Analysis", "K-means Clustering", "Advanced Graph"]

# Load data chips from Milvus
def load_data_chips():
    try:
        results = client.query(
            collection_name="text_embeddings",
            expr="",
            output_fields=["id"],
            limit=100  # Add a limit to the query
        )
        return [result["id"] for result in results]
    except Exception as e:
        st.error(f"Failed to load data chips: {e}")
        return []

# Update session state with data chips from Milvus
data_chips = load_data_chips()

# Combine data chips and standard chips
all_chips = standard_chips + data_chips

# Drag-and-drop chips interface
selected_chips = st_tags(
    label='Drag and drop chips into the query field:',
    text='Press enter to add more',
    value=[],
    suggestions=all_chips,
    maxtags=10,
    key='1'
)

# Chatbot interface for querying embeddings
if 'all_texts' in st.session_state:
    st.write("### Chat with the Data")
    user_query = st.text_input("Ask a question about the data or request a graph:")
    
    if st.button("Submit Query"):
        if any(chip in selected_chips for chip in data_chips):
            st.write("Data has been uploaded.")
        
        if "Sentiment Analysis" in selected_chips:
            st.write("Performing sentiment analysis on the data...")
            sentiments = perform_sentiment_analysis(st.session_state['all_texts'])
            st.write(sentiments)
        
        if "K-means Clustering" in selected_chips:
            st.write("Performing K-means clustering on the data...")
            clusters = perform_kmeans_clustering(st.session_state['all_texts'])
            st.write(clusters)
        
        if "Advanced Graph" in selected_chips:
            st.write("Generating advanced graph based on the data...")
            data = pd.DataFrame({
                'x': range(10),
                'y': range(10),
                'label': ['A']*5 + ['B']*5
            })
            generate_advanced_graph(data, graph_type="scatter")
        
        if user_query:
            # Query the data using GPT-4
            combined_text = ' '.join(st.session_state['all_texts'])
            text_chunks = list(chunk_text(combined_text))
            responses = []
            for chunk in text_chunks:
                response = generate_relevant_response(chunk, user_query)
                responses.append(response)
            full_response = " ".join(responses)
            
            # Summarize the full response
            summarized_response = summarize_text(full_response)
            st.write(summarized_response)
        
        # Embedding search query
        search_results = search_embeddings(user_query)
        st.write(search_results)


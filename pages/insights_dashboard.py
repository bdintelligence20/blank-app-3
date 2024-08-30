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
import nltk
from streamlit_tags import st_tags
import re

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

# Function to preprocess text data using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Function to chunk text into manageable sizes for the model
def chunk_text(text, chunk_size=2000):
    sentences = text.split('. ')
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence.split()) > chunk_size:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence.split())
    if current_chunk:
        yield ' '.join(current_chunk)

# Function to generate a comprehensive and relevant response using GPT-4
def generate_relevant_response(data, query):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant that provides concise and accurate answers to the user's questions based on the data provided."},
            {"role": "user", "content": f"Based on the following data: {data}. {query}"}
        ],
        temperature=0.3,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0.30,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Function to handle simple queries using a smaller model
def handle_simple_query(text, query):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that provides specific information from the text."},
            {"role": "user", "content": f"Based on the following text: {text}. {query}"}
        ],
        temperature=0.3,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

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

# Drag-and-drop chips interface
selected_chips = st_tags(
    label='Drag and drop chips into the query field:',
    text='Press enter to add more',
    value=[],
    suggestions=standard_chips,
    maxtags=10,
    key='1'
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Provide a dropdown to select a document
if 'document_metadata' in st.session_state:
    document_options = [f"{doc['source']} ({doc['type']})" for doc in st.session_state['document_metadata']]
    selected_document = st.selectbox("Select a document to query:", document_options)
else:
    st.error("No documents available. Please upload data on the data page.")

# Accept user input
if prompt := st.chat_input("Ask a question about the selected document:"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the query based on selected document
    if 'all_texts' in st.session_state and selected_document:
        selected_index = document_options.index(selected_document)
        selected_text = st.session_state['all_texts'][selected_index]

        # Handle simple queries using a smaller model
        simple_response = handle_simple_query(selected_text, prompt)
        if simple_response:
            st.write(simple_response)
        else:
            # Process the query based on selected chips
            if "Sentiment Analysis" in selected_chips:
                st.write("Performing sentiment analysis on the data...")
                sentiments = perform_sentiment_analysis([selected_text])
                st.write(sentiments)
            
            if "K-means Clustering" in selected_chips:
                st.write("Performing K-means clustering on the data...")
                clusters = perform_kmeans_clustering([selected_text])
                st.write(clusters)
            
            if "Advanced Graph" in selected_chips:
                st.write("Generating advanced graph based on the data...")
                data = pd.DataFrame({
                    'x': range(10),
                    'y': range(10),
                    'label': ['A']*5 + ['B']*5
                })
                generate_advanced_graph(data, graph_type="scatter")
            
            # Query the data using GPT-4
            text_chunks = list(chunk_text(selected_text))
            responses = []
            for chunk in text_chunks:
                response = generate_relevant_response(chunk, prompt)
                responses.append(response)
            full_response = " ".join(responses)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("No data available. Please upload data on the data page.")

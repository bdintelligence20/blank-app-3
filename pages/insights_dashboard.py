import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from openai import OpenAI
import spacy
from pymilvus import MilvusClient

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

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
    results = client.search(
        collection_name="text_embeddings",
        data=[query_embedding],
        anns_field="vector",
        params=search_params,
        limit=top_k,
        output_fields=["vector"]
    )
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
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

# Store data and allow querying through a chatbot interface
st.title("Interactive Chatbot for Data Analysis")

# Chatbot interface for querying embeddings
if 'all_texts' in st.session_state:
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
        st.write("Search results for embeddings:")
        st.write(search_results)

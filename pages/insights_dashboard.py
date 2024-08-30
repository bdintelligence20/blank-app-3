import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_parse import LlamaParse
import pandas as pd
import PyPDF2
import docx
import os
import requests
from bs4 import BeautifulSoup

# Set up Streamlit page
st.title("RAG Pipeline with LlamaIndex and LlamaParse")
st.write("Upload documents or scrape websites to create and query an index using an LLM.")

# Fetch API keys from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
llama_cloud_api_key = st.secrets["llama_cloud"]["api_key"]

# Set the OpenAI API key for the OpenAI client
import openai
openai.api_key = openai_api_key

# Set up OpenAI LLM with specified model and temperature
Settings.llm = OpenAI(api_key=openai_api_key, temperature=0.2, model="gpt-4")

# Set your Llama Cloud API key
os.environ['LLAMA_CLOUD_API_KEY'] = llama_cloud_api_key

# Directory to persist the index
PERSIST_DIR = "index_storage"

# Check if the persisted index exists and load it
if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    st.session_state.index = load_index_from_storage(storage_context)
    st.write("Loaded persisted index from disk.")
else:
    # Initialize a new index in session state
    if 'index' not in st.session_state:
        st.session_state.index = VectorStoreIndex()

# Function to process different file types into Document objects
def load_document_from_file(file):
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
        text = df.to_string()
        return Document(text=text)
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        text = df.to_string()
        return Document(text=text)
    elif file.type == "application/pdf":
        # Use LlamaParse for better PDF parsing
        parser = LlamaParse(result_type="markdown")
        documents = parser.load_data(file)
        return documents
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return Document(text=text)
    else:
        return None

# Function to scrape content from a webpage
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract all text from the webpage
        text = ' '.join(soup.stripped_strings)
        return Document(text=text)
    except Exception as e:
        st.write(f"Failed to scrape {url}: {e}")
        return None

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload Excel, CSV, PDF, or Word files", type=["xlsx", "csv", "pdf", "docx"], accept_multiple_files=True)

# User input for URLs to scrape
url_input = st.text_area("Enter URLs to scrape data (one per line):")
urls = [url.strip() for url in url_input.splitlines() if url.strip()]

# Process and index documents when files are uploaded or URLs are provided
if st.button("Index Uploaded Documents and Scrape Websites"):
    documents = []
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            docs = load_document_from_file(file)
            if docs:
                if isinstance(docs, list):
                    documents.extend(docs)  # LlamaParse returns a list of documents
                else:
                    documents.append(docs)
    
    # Scrape websites
    if urls:
        for url in urls:
            doc = scrape_website(url)
            if doc:
                documents.append(doc)
    
    if documents:
        st.session_state.index.add_documents(documents)
        # Persist the index to disk
        st.session_state.index.storage_context.persist(persist_dir=PERSIST_DIR)
        st.write("Documents and websites have been indexed and persisted to disk.")
    else:
        st.write("No valid documents or websites provided for indexing.")

# Initialize QueryEngine with streaming enabled
query_engine = st.session_state.index.as_query_engine(streaming=True)

# User input for query
user_query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if user_query:
        # Perform the query using the QueryEngine with streaming
        streaming_response = query_engine.query(user_query)

        # Create a placeholder for streaming output
        response_placeholder = st.empty()

        # Stream the response as it is generated
        response_text = ""
        for text in streaming_response.response_gen:
            response_text += text
            response_placeholder.text(response_text)  # Update the text in the placeholder
    else:
        st.write("Please enter a query.")

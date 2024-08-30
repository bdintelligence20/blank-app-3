import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_parse import LlamaParse
import pandas as pd
import PyPDF2
import docx
import os
import requests
from bs4 import BeautifulSoup

# Set up Streamlit page
st.title("Enhanced RAG Pipeline with LlamaIndex and LlamaParse")
st.write("Upload documents or scrape websites to create and query an index using an LLM.")

# Fetch API keys from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
llama_cloud_api_key = st.secrets["llama_cloud"]["api_key"]

# Set the OpenAI API key for the OpenAI client
import openai
openai.api_key = openai_api_key

# Set up OpenAI LLM with specified model and temperature
Settings.llm = OpenAI(api_key=openai_api_key, temperature=0.2, model="gpt-4o", max_tokens=4095)

# Set your Llama Cloud API key
os.environ['LLAMA_CLOUD_API_KEY'] = llama_cloud_api_key

# Directory to persist the index
PERSIST_DIR = "index_storage"

# Check if the persisted index exists and load it
if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    st.session_state.index = load_index_from_storage(storage_context)
    st.session_state.documents = st.session_state.index.documents  # Load documents from index
    st.write("Loaded persisted index from disk.")
else:
    # Initialize an empty index or a new one with documents later
    st.session_state.index = None
    st.session_state.documents = []  # Initialize empty documents list

# Function to process different file types into Document objects using LlamaParse
def load_document_from_file(file):
    parser = LlamaParse(result_type="markdown")
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
        text = df.to_string()
        return [Document(text=text)]
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        text = df.to_string()
        return [Document(text=text)]
    elif file.type == "application/pdf":
        # Use LlamaParse for better PDF parsing
        return parser.load_data(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return [Document(text=text)]
    else:
        return None

# Function to scrape content from a webpage and parse using LlamaParse
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract all text from the webpage
        text = ' '.join(soup.stripped_strings)
        return [Document(text=text)]
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
    new_documents = []
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            docs = load_document_from_file(file)
            if docs:
                new_documents.extend(docs)  # LlamaParse returns a list of documents
    
    # Scrape websites
    if urls:
        for url in urls:
            docs = scrape_website(url)
            if docs:
                new_documents.extend(docs)
    
    if new_documents:
        # Add new documents to session state
        st.session_state.documents.extend(new_documents)
        
        # Recreate the index with all documents
        st.session_state.index = VectorStoreIndex.from_documents(st.session_state.documents)
        
        # Persist the index to disk
        st.session_state.index.storage_context.persist(persist_dir=PERSIST_DIR)
        st.write("Documents and websites have been indexed and persisted to disk.")
    else:
        st.write("No valid documents or websites provided for indexing.")

# Configure a comprehensive response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="refine")  # Use "refine" for detailed answers

# Initialize QueryEngine with streaming enabled if the index is not None
if st.session_state.index is not None:
    query_engine = st.session_state.index.as_query_engine(response_synthesizer=response_synthesizer, streaming=True)

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
                response_placeholder.markdown(response_text)  # Update the text in the placeholder

            # Optionally, add a summary or detailed explanation step
            if st.button("Get Summary and Detailed Explanation"):
                detailed_prompt = f"Provide a detailed explanation and summary for the following: {response_text}"
                detailed_response = query_engine.query(detailed_prompt)
                st.write("Detailed Response:", detailed_response)
        else:
            st.write("Please enter a query.")

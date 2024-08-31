import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import os
import pandas as pd
import re

# Database setup
engine = create_engine('sqlite:///knowledge_base.db')
Base = declarative_base()

class DocumentEntry(Base):  # Renamed to avoid confusion with LlamaIndex Document class
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    content = Column(Text)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def add_document(name, content):
    # Check if the document already exists
    existing_doc = session.query(DocumentEntry).filter_by(name=name).first()
    if existing_doc:
        # Update the existing document's content
        existing_doc.content = content
        session.commit()
        return

    # If document doesn't exist, add a new one
    doc = DocumentEntry(name=name, content=content)
    session.add(doc)
    session.commit()

def get_documents():
    return session.query(DocumentEntry).all()

# Helper function to split large text into chunks
def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Streamlit app setup
st.set_page_config(
    page_title="Chat with the Streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

openai.api_key = st.secrets["openai"]["api_key"]
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # Load documents from the database and convert them to LlamaIndex Document objects
    docs = [
        Document(
            text=doc.content,
            metadata={"filename": doc.name},
            doc_id=str(doc.id)
        ) 
        for doc in get_documents()
    ]
    
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=16383,  # Reduced to ensure responses fit within the limit
        top_p=1,
        system_prompt="""You are a helpful assistant."""
    )
    
    index = VectorStoreIndex.from_documents(docs)  # Use the list of Document objects
    return index

# Add file upload functionality
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "xlsx", "csv"])

if uploaded_file is not None:
    file_content = ""

    # Handle different file types
    if uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        st.error("PDF processing not implemented yet. Please upload a text, CSV, or Excel file.")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        file_content = df.to_string()
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        file_content = df.to_string()
    else:
        st.error("Unsupported file type.")

    if file_content:
        # Save uploaded file content to database
        add_document(name=uploaded_file.name, content=file_content)
        st.success(f"Document {uploaded_file.name} added to the knowledge base.")

index = load_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Adjust the prompt to include a specific request for a complete list
    modified_prompt = f"{prompt}. Please provide a complete list without truncation."
    st.session_state.messages[-1]["content"] = modified_prompt

    response_text = ""
    if len(prompt) > 1000:  # If the document is too large, split it into chunks
        chunks = split_text(prompt)
        for chunk in chunks:
            response_stream = st.session_state.chat_engine.stream_chat(chunk)
            response_text += response_stream.response
    else:
        response_stream = st.session_state.chat_engine.stream_chat(modified_prompt)
        response_text = response_stream.response

    # Validate and consolidate all extracted details
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response_text)
    if emails:
        response_text = "\n".join(set(emails))  # Removing duplicates

    st.write(response_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        st.session_state.messages.append(message)

import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import pandas as pd

# Database setup
engine = create_engine('sqlite:///knowledge_base.db')
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    content = Column(Text)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

def add_document(name, content):
    doc = Document(name=name, content=content)
    session.add(doc)
    session.commit()

def get_documents():
    return session.query(Document).all()

# Streamlit app setup
st.set_page_config(
    page_title="Chat with the Streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto"
)

openai.api_key = st.secrets.openai_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!",
        }
    ]

@st.cache_resource(show_spinner=True)
def load_data():
    # Load documents from the database
    docs = [Document(name=doc.name, content=doc.content) for doc in get_documents()]
    
    # Broaden the system prompt for more comprehensive answers
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are a highly knowledgeable assistant skilled in 
        various domains of software development and data science. While your 
        primary focus is to provide technical assistance regarding the Streamlit 
        Python library, you are also equipped to offer insights and best practices 
        in related areas such as web development, machine learning, data visualization, 
        and Python programming in general. When appropriate, extend your responses to 
        include broader concepts and useful tips beyond the Streamlit documentation, 
        ensuring answers are informative and based on well-established knowledge.""",
    )
    
    index = VectorStoreIndex.from_documents([doc.content for doc in docs])
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        st.session_state.messages.append(message)

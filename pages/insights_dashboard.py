import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from openai import OpenAI

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI client
client = OpenAI()

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            return pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/json":
            return pd.read_json(uploaded_file)
        elif uploaded_file.type == "text/plain":
            return pd.read_csv(uploaded_file, delimiter='\t')
    return None

# Function to preprocess text
def preprocess_text(text, lower=True, remove_stopwords=True, lemmatize=True):
    doc = nlp(text)
    if lower:
        text = text.lower()
    tokens = [token.text for token in doc]
    if remove_stopwords:
        tokens = [token for token in tokens if not nlp.vocab[token].is_stop]
    if lemmatize:
        tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    return " ".join(tokens)

# Function to perform sentiment analysis using TextBlob
def sentiment_analysis_textblob(text):
    return TextBlob(text).sentiment.polarity

# Function to perform sentiment analysis using VADER
def sentiment_analysis_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

# Function to generate word cloud
def generate_wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Function to perform topic modeling
def topic_modeling(texts, n_topics=5, vectorizer_type="tfidf"):
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    else:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    return lda, vectorizer

# Function to display LDA topics
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx+1}"] = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return pd.DataFrame(topics.items(), columns=["Topic", "Top Words"])

# Function to perform NER and extract entities
def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to perform clustering and dimensionality reduction
def clustering_and_dimensionality_reduction(texts, n_clusters=3, method='pca'):
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts).toarray()
    
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2)
    
    X_reduced = reducer.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)
    
    df = pd.DataFrame(X_reduced, columns=['Component 1', 'Component 2'])
    df['Cluster'] = clusters
    return df

# Function to query the LLM
def query_llm(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1,
        max_tokens=16383,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"}
    )
    return response.choices[0].message['content']

# Streamlit App
st.title("Ultra-Advanced Qualitative Data Analysis Tool with LLM Assistant")

# File upload
uploaded_file = st.file_uploader("Upload your qualitative data file (CSV, JSON, or TXT)", type=["csv", "json", "txt"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("### Uploaded Data")
        st.write(data.head())
        
        # Choose the text column for analysis
        text_column = st.selectbox("Select the column containing text data", data.columns)
        
        # Preprocessing options
        lower = st.checkbox("Convert to lowercase", value=True)
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        lemmatize = st.checkbox("Lemmatize", value=True)
        
        # Preprocess text data
        data['processed_text'] = data[text_column].apply(lambda x: preprocess_text(x, lower, remove_stopwords, lemmatize))
        
        # Word Cloud
        st.write("### Word Cloud")
        all_text = " ".join(data['processed_text'])
        generate_wordcloud(all_text)
        
        # Sentiment Analysis
        st.write("### Sentiment Analysis")
        data['sentiment_textblob'] = data['processed_text'].apply(sentiment_analysis_textblob)
        data['sentiment_vader'] = data['processed_text'].apply(sentiment_analysis_vader)
        st.write(data[['processed_text', 'sentiment_textblob', 'sentiment_vader']].head())
        
        # Sentiment Distribution
        st.write("### Sentiment Distribution")
        fig = px.histogram(data, x='sentiment_textblob', title='TextBlob Sentiment Polarity Distribution', nbins=20)
        st.plotly_chart(fig)
        
        fig = px.histogram(data, x='sentiment_vader', title='VADER Sentiment Polarity Distribution', nbins=20)
        st.plotly_chart(fig)
        
        # Named Entity Recognition
        st.write("### Named Entity Recognition")
        entity_sample = data['processed_text'].sample(1).values[0]
        entities = named_entity_recognition(entity_sample)
        st.write(f"Entities in Sample Text: {entity_sample}")
        st.write(entities)
        
        # Topic Modeling
        st.write("### Topic Modeling")
        n_topics = st.slider("Select number of topics", 2, 10, 5)
        vectorizer_type = st.selectbox("Vectorizer type", ["tfidf", "count"])
        lda_model, vectorizer = topic_modeling(data['processed_text'], n_topics, vectorizer_type)
        topics_df = display_topics(lda_model, vectorizer.get_feature_names_out(), 10)
        st.write(topics_df)
        
        # Clustering and Dimensionality Reduction
        st.write("### Clustering and Dimensionality Reduction")
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        method = st.selectbox("Dimensionality reduction method", ["pca", "tsne"])
        cluster_df = clustering_and_dimensionality_reduction(data['processed_text'], n_clusters, method)
        fig = px.scatter(cluster_df, x='Component 1', y='Component 2', color='Cluster', title='Clustering and Dimensionality Reduction')
        st.plotly_chart(fig)

        # Text Classification
        st.write("### Text Classification")
        if 'label' in data.columns:
            X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['label'], test_size=0.2, random_state=42)
            vectorizer = TfidfVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            model = MultinomialNB()
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            
            st.write("Classification Report")
            st.text(classification_report(y_test, y_pred))
            
            st.write("Confusion Matrix")
            fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig)
        
        # LLM Assistant
        st.write("### LLM Data Assistant")
        user_question = st.text_input("Ask a question about the data", placeholder="Enter your question here...")

        # If a question is provided by the user
        if user_question:
            # Preparing messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a data assistant capable of analyzing qualitative data and answering questions based on the dataset provided."
                },
                {
                    "role": "user",
                    "content": f"The data provided has the following text column: {text_column}. Please analyze this data. {user_question}"
                }
            ]
            
            # Query the LLM
            with st.spinner("Analyzing your question..."):
                response_text = query_llm(messages)
                
            # Display the response from the LLM
            st.write("### Assistant Response")
            st.write(response_text)
    
    else:
        st.error("Failed to load the data. Please check the file format.")

else:
    st.info("Please upload a qualitative data file for analysis.")

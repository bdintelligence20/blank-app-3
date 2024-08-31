import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from flair.data import Sentence
from flair.models import SequenceTagger

# Initialize OpenAI client using Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

# Function to load data
def load_data(uploaded_files):
    dataframes = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/json":
            df = pd.read_json(uploaded_file)
        elif uploaded_file.type == "text/plain":
            df = pd.read_csv(uploaded_file, delimiter='\t')
        else:
            st.error("Unsupported file type!")
            return None
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    else:
        return ""

# Function to perform sentiment analysis using TextBlob
def sentiment_analysis_textblob(text):
    return TextBlob(text).sentiment.polarity

# Function to perform sentiment analysis using VADER
def sentiment_analysis_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

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

# Function to generate heatmap for topics
def generate_topic_heatmap(dtm, lda_model):
    topic_dist = lda_model.transform(dtm)
    sns.heatmap(topic_dist, annot=False, cmap='coolwarm')
    st.pyplot(plt)

# Function to perform named entity recognition using Flair
def named_entity_recognition(text):
    sentence = Sentence(text)
    tagger = SequenceTagger.load("ner")
    tagger.predict(sentence)
    entities = [(entity.text, entity.get_label("ner").value) for entity in sentence.get_spans("ner")]
    return entities

# Streamlit App
st.title("Qualitative Data Analysis Dashboard")

# Sidebar for file uploads
st.sidebar.header("Upload Your Data Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload your qualitative data files (CSV, JSON, TXT)", 
    type=["csv", "json", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    data = load_data(uploaded_files)
    if data is not None:
        st.sidebar.subheader("Select Columns for Analysis")
        text_columns = st.sidebar.multiselect(
            "Select the text columns you want to analyze", 
            data.columns
        )

        if text_columns:
            # Preprocess text data
            data['processed_text'] = data[text_columns].astype(str).apply(lambda x: ' '.join(x), axis=1).apply(preprocess_text)
            
            st.header("Data Analysis")

            # Word Cloud
            st.subheader("Word Cloud")
            all_text = " ".join(data['processed_text'])
            generate_wordcloud(all_text)
            
            # Sentiment Analysis
            st.subheader("Sentiment Analysis")
            data['sentiment'] = data['processed_text'].apply(sentiment_analysis_vader)
            data['positive'] = data['sentiment'].apply(lambda x: x['pos'])
            data['neutral'] = data['sentiment'].apply(lambda x: x['neu'])
            data['negative'] = data['sentiment'].apply(lambda x: x['neg'])
            sentiment_counts = data[['positive', 'neutral', 'negative']].mean()
            st.bar_chart(sentiment_counts)
            
            # Topic Modeling
            st.subheader("Topic Modeling")
            n_topics = st.slider("Select number of topics", 2, 10, 5)
            vectorizer_type = st.selectbox("Vectorizer type", ["tfidf", "count"])
            lda_model, vectorizer = topic_modeling(data['processed_text'], n_topics, vectorizer_type)
            topics_df = display_topics(lda_model, vectorizer.get_feature_names_out(), 10)
            st.write(topics_df)
            
            # Topic Heatmap
            st.subheader("Topic Heatmap")
            dtm = vectorizer.transform(data['processed_text'])
            generate_topic_heatmap(dtm, lda_model)
        
        else:
            st.warning("Please select at least one column for analysis.")
    else:
        st.error("Failed to load the data. Please check the file format.")
else:
    st.sidebar.info("Please upload qualitative data files for analysis.")

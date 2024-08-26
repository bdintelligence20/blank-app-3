import os
import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text data
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = ' '.join([w for w in word_tokens if w.isalpha() and w not in stop_words and w not in string.punctuation])
    return filtered_text

# Access the API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to generate insights using GPT-4
def generate_insights(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert analyst."},
                  {"role": "user", "content": f"Analyze the following text and provide 10 common problems: {text}"}],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"}
    )
    return response.choices[0].message['content']

# Streamlit UI
st.title("Text Analysis with GPT-4")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load Excel data
    @st.cache_data
    def load_data(file):
        data = pd.read_excel(file)
        data.fillna("", inplace=True)  # Fill NaN values with empty strings
        return data

    data = load_data(uploaded_file)

    # Display division options and filter data
    division_options = data['Division (TD, TT, TA, Impactful)'].unique()
    selected_division = st.selectbox("Select a Division:", division_options)
    filtered_data = data[data['Division (TD, TT, TA, Impactful)'] == selected_division]

    # Combine all relevant columns into one
    filtered_data['All_Problems'] = filtered_data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    filtered_data['Processed_Text'] = filtered_data['All_Problems'].apply(preprocess_text)

    # Perform text vectorization using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(filtered_data['Processed_Text'])

    # Perform KMeans clustering
    num_clusters = st.slider('Select number of clusters:', 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    filtered_data['Cluster'] = kmeans.labels_

    # Display clusters
    for cluster_num in range(num_clusters):
        st.write(f"### Cluster {cluster_num + 1}")
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster_num]['All_Problems'].tolist()
        insights = generate_insights(' '.join(cluster_data))
        st.write(insights)

    # Display the processed data
    st.write(filtered_data)

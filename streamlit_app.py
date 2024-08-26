import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set the NLTK data path to where the resources are downloaded
nltk.data.path.append('/home/vscode/nltk_data')

# Function to preprocess text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# Streamlit app layout
st.title("LRMG Problem Analysis Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Load Excel data
    @st.cache_data
    def load_data(file):
        data = pd.read_excel(file)
        data.fillna("", inplace=True)  # Fill NaN values with empty strings for easier processing
        return data

    data = load_data(uploaded_file)

    # Display division options and filter data
    division_options = data['Division (TD, TT, TA, Impactful)'].unique()
    selected_division = st.selectbox('Select Division:', division_options)
    filtered_data = data[data['Division (TD, TT, TA, Impactful)'] == selected_division]

    st.subheader("Filtered Data by Division")
    st.write(filtered_data)

    # Combine all problem descriptions into one text column for analysis
    filtered_data['All_Problems'] = filtered_data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Preprocess the text data
    filtered_data['Processed_Text'] = filtered_data['All_Problems'].apply(preprocess_text)

    # Perform text vectorization using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(filtered_data['Processed_Text'])

    # Perform KMeans clustering
    num_clusters = st.slider('Select number of clusters:', 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    filtered_data['Cluster'] = kmeans.fit_predict(X)

    st.subheader("Clustered Data")
    st.write(filtered_data[['Division (TD, TT, TA, Impactful)', 'All_Problems', 'Cluster']])

    # Display common themes in each cluster
    st.subheader("Common Themes in Clusters")
    for cluster in range(num_clusters):
        st.write(f"Cluster {cluster}:")
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
        cluster_text = ' '.join(cluster_data['Processed_Text'])
        word_tokens = word_tokenize(cluster_text)
        word_freq = pd.Series(word_tokens).value_counts().head(10)
        st.write(word_freq)
else:
    st.warning("Please upload an Excel file to proceed.")

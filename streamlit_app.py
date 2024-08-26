import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text data using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    filtered_words = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(filtered_words)

# Function to extract top terms per cluster
def get_top_terms_per_cluster(vectorizer, kmeans, num_terms=10):
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    top_terms = {}
    for i in range(kmeans.n_clusters):
        top_terms[i] = [terms[ind] for ind in order_centroids[i, :num_terms]]
    return top_terms

# Function to provide qualitative analysis based on top terms
def provide_qualitative_analysis(top_terms):
    analysis = {}
    for cluster, terms in top_terms.items():
        analysis[cluster] = f"Common problems for this cluster seem to involve aspects such as {', '.join(terms[:5])}."
    return analysis

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
    vectorizer = TfidfVectorizer(stop_words=sklearn_stop_words)
    X = vectorizer.fit_transform(filtered_data['Processed_Text'])

    # Perform KMeans clustering
    num_clusters = st.slider('Select number of clusters:', 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    filtered_data['Cluster'] = kmeans.fit_predict(X)

    st.subheader("Clustered Data")
    st.write(filtered_data[['Division (TD, TT, TA, Impactful)', 'All_Problems', 'Cluster']])

    # Extract top terms per cluster for analysis
    top_terms = get_top_terms_per_cluster(vectorizer, kmeans)

    # Provide qualitative analysis based on top terms
    qualitative_analysis = provide_qualitative_analysis(top_terms)

    st.subheader("Qualitative Analysis of Clusters")
    for cluster, analysis in qualitative_analysis.items():
        st.write(f"Cluster {cluster}: {analysis}")

    # Display top terms for each cluster
    st.subheader("Top Terms in Each Cluster")
    for cluster, terms in top_terms.items():
        st.write(f"Cluster {cluster}: {', '.join(terms)}")

    # Display common problems solved based on clusters
    st.subheader("Common Problems Solved")
    common_problems = []
    for cluster, terms in top_terms.items():
        problem = f"Cluster {cluster} primarily deals with problems related to: {', '.join(terms[:5])}."
        common_problems.append(problem)
        st.write(problem)
else:
    st.warning("Please upload an Excel file to proceed.")

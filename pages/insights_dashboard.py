import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Debug function to print session state keys
def print_session_state():
    st.write("### Session State Keys")
    st.write(list(st.session_state.keys()))

# Function to display top 5 problems per division based on problem-solving questionnaire
def display_top_problems_per_division():
    st.subheader("Top 5 Problems Per Division Based on Problem Solving Questionnaire")
    divisions = ['Talent Development', 'Talent Technology', 'Talent Advisory', 'Africa', 'Corporate', 'Marketing']
    for division in divisions:
        if f'{division}_problems' in st.session_state:
            st.write(f"**{division}**")
            problems = st.session_state[f'{division}_problems']
            for problem in problems:
                st.write(f"- {problem}")
            st.write("")
        else:
            st.write(f"No data available for {division}")

# Function to display consolidated clusters for all divisions
def display_consolidated_clusters():
    if 'consolidated_clusters' in st.session_state:
        st.subheader("Consolidated Clusters (All Divisions)")
        clusters = st.session_state['consolidated_clusters']
        for idx, cluster in enumerate(clusters):
            st.write(f"**Cluster {idx + 1}**")
            st.write(cluster)
            st.write("")
    else:
        st.write("No consolidated clusters data available.")

# Function to display consolidated data visualizations
def display_data_visualizations(data, title):
    st.subheader(f"Consolidated Data Visualizations ({title})")
    if data is not None and not data.empty:
        st.write("**Keyword Frequency**")
        fig, ax = plt.subplots()
        sns.barplot(x='Count', y='Keyword', data=data, ax=ax)
        ax.set_xlabel('Count')
        ax.set_ylabel('Keyword')
        st.pyplot(fig)
    else:
        st.write("No keyword data available.")

# Function to display keyword data
def display_keywords(short_tail, long_tail, title):
    st.subheader(f"{title} Keywords")
    st.write("**Short-Tail Keywords**")
    if short_tail is not None and not short_tail.empty:
        st.write(short_tail)
    else:
        st.write("No short-tail keywords available.")
    
    st.write("**Long-Tail Keywords**")
    if long_tail is not None and not long_tail.empty:
        st.write(long_tail)
    else:
        st.write("No long-tail keywords available.")

st.title("Insights Dashboard")

# Print session state for debugging
print_session_state()

# Check if Excel data insights are available in session state
if 'excel_insights' in st.session_state:
    # Data sheet upload insights
    st.header("Data Sheet Upload")
    
    # Top 5 problems per division
    display_top_problems_per_division()

    # Consolidated clusters (All divisions)
    display_consolidated_clusters()

    # Consolidated data visualizations (All divisions)
    if 'excel_keyword_data' in st.session_state:
        display_data_visualizations(st.session_state['excel_keyword_data'], "All Divisions")

    # Display keywords for Excel data
    if 'excel_short_tail_keywords' in st.session_state and 'excel_long_tail_keywords' in st.session_state:
        display_keywords(st.session_state['excel_short_tail_keywords'], st.session_state['excel_long_tail_keywords'], "Excel Data")

# Check if web scraping insights are available in session state
if 'web_insights' in st.session_state:
    st.header("Web Scraping")

    # Top 10 problems LRMG solves based on website
    if 'web_problems' in st.session_state:
        st.subheader("Top 10 Problems LRMG Solves Based on Website")
        for problem in st.session_state['web_problems']:
            st.write(f"- {problem}")

    # Clusters based on website
    if 'web_clusters' in st.session_state:
        st.subheader("Clusters Based on Website")
        display_consolidated_clusters()

    # Consolidated data visualizations for web scraping
    if 'web_keyword_data' in st.session_state:
        display_data_visualizations(st.session_state['web_keyword_data'], "Web Scraping")

    # Display keywords for web data
    if 'web_short_tail_keywords' in st.session_state and 'web_long_tail_keywords' in st.session_state:
        display_keywords(st.session_state['web_short_tail_keywords'], st.session_state['web_long_tail_keywords'], "Web Data")

# Message if no data is available
if 'excel_insights' not in st.session_state and 'web_insights' not in st.session_state:
    st.write("No data available. Please upload data on the Data Page.")

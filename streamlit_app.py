import streamlit as st

# Set page configuration
st.set_page_config(page_title="Multi-Page Streamlit App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Data Page", "Insights Dashboard"])

if page == "Welcome":
    st.title("Welcome to the Multi-Page Streamlit App")
    st.write("Use the navigation options in the sidebar to access different pages of the app.")
    st.write("Explore data, gain insights, and make data-driven decisions with our powerful analytics tools.")
    
elif page == "Data Page":
    st.title("Data Page")
    st.write("Upload and analyze your data here.")
    # Include the content from your current data page script here
    # This can be done by calling the data page script directly or embedding it here.

    # Example placeholder for data page:
    exec(open("pages/data_page.py").read())

elif page == "Insights Dashboard":
    st.title("Insights Dashboard")
    st.write("This page will display insights and visualizations.")
    # Include the content from your insights dashboard script here.

    # Example placeholder for insights dashboard:
    exec(open("pages/insights_dashboard.py").read())

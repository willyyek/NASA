import streamlit as st

st.set_page_config(page_title="Test Sidebar", layout="wide")

st.sidebar.title("🔭 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Novice Mode", "Researcher Mode"])

st.write("当前页面:", page)

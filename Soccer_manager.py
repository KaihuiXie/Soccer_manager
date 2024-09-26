import streamlit as st
from footer import footer


st.set_page_config(
    page_title="Soccer Manager",
    page_icon="⚽",
)
footer()

st.write("# Welcome to Soccer Manager! 👋")

st.markdown("""
## What you can do
- Analyze your own matches
- Review past matches in different leagues
- Tune your own matches
""")


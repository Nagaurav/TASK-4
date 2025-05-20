import streamlit as st
from transformers import pipeline

st.title("RAG App with Hugging Face Token")

# Load HF token from secrets
hf_token = st.secrets["hf_token"]

st.write("Using Hugging Face token securely from Streamlit secrets.")

# Initialize pipeline with token
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    use_auth_token=hf_token
)

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text.strip():
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        st.success("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

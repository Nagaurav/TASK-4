import streamlit as st
from rag_utils import add_documents, query_rag

st.set_page_config(page_title="Chat with Your Documents", layout="centered")
st.title("📄 Chat with Your Documents using RAG")

st.subheader("Step 1: Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if uploaded_files and st.button("Embed Documents"):
    with st.spinner("Processing documents..."):
        msg = add_documents(uploaded_files)
        st.success(msg)

st.subheader("Step 2: Ask a Question")
user_query = st.text_input("What would you like to know?")

if user_query:
    with st.spinner("Searching..."):
        answer = query_rag(user_query)
        st.info(answer)

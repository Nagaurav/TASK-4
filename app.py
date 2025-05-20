import streamlit as st
from rag_utils import add_documents, query_rag

st.title("📄 Chat with Your Documents using RAG")

# Upload section
st.header("Step 1: Upload documents")
files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if files and st.button("Embed Documents"):
    with st.spinner("Processing..."):
        msg = add_documents(files)
        st.success(msg)

# Query section
st.header("Step 2: Ask Questions")
question = st.text_input("Ask a question about your uploaded documents")

if question:
    with st.spinner("Thinking..."):
        response = query_rag(question)
        st.info(response)

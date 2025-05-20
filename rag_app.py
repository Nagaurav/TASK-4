import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF
import PyPDF2
import docx
import os

# Load Hugging Face token securely
hf_token = st.secrets["hf_token"]

@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", use_auth_token=hf_token)

qa_pipeline = load_model()

st.title("📄 RAG Resume Q&A App")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT Resume", type=["pdf", "docx", "txt"])
question = st.text_input("Ask a question about the resume:")

def extract_text(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == "docx":
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        return str(file.read(), "utf-8")
    return ""

if uploaded_file and question:
    with st.spinner("Processing..."):
        context = extract_text(uploaded_file)
        result = qa_pipeline(question=question, context=context)
        st.markdown(f"**Answer:** {result['answer']}")

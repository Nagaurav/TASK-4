import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import docx
from io import StringIO

st.set_page_config(page_title="RAG with Hugging Face", layout="wide")

# Load Hugging Face API key
HF_TOKEN = st.secrets["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"]
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Hugging Face QA
def ask_question(question, context):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("answer", "No answer found.")
    else:
        return "Failed to get answer from Hugging Face."

# Document Parsing
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return StringIO(file.getvalue().decode()).read()
    return ""

# Chunking
def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Embedding
@st.cache_resource
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# UI
st.title("📚 Chat With Your Documents (Hugging Face RAG)")
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        all_text += extract_text(file) + "\n"

    text_chunks = split_text(all_text)
    index, embeddings, chunks = embed_chunks(text_chunks)

    st.success("Documents processed. Ask your question below.")

    user_query = st.text_input("Ask a question about the documents:")
    if user_query:
        query_embed = SentenceTransformer("all-MiniLM-L6-v2").encode([user_query])
        D, I = index.search(query_embed, k=3)
        top_chunks = "\n".join([chunks[i] for i in I[0]])
        answer = ask_question(user_query, top_chunks)
        st.markdown(f"**Answer:** {answer}")

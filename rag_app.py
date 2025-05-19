import streamlit as st
import os
import tempfile

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import docx
import fitz  # PyMuPDF

# Load Hugging Face API token
HF_TOKEN = st.secrets["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"]
os.environ["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"] = HF_TOKEN

st.title("📄 Chat with Your Documents (RAG App)")

# File uploader
uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    rag_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    return embedder, rag_pipeline

embedder, rag_pipeline = load_models()

# Document parsing
def parse_file(file):
    text = ""
    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "text/plain":
        text = file.read().decode()
    return text

# Embed documents and build FAISS index
def build_vector_store(text_chunks):
    embeddings = embedder.encode(text_chunks)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Chunking helper
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Process uploaded files
all_chunks = []
file_sources = []
if uploaded_files:
    for file in uploaded_files:
        raw_text = parse_file(file)
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
        file_sources.extend([file.name] * len(chunks))

    index, embeddings = build_vector_store(all_chunks)
    st.success(f"{len(all_chunks)} chunks indexed from {len(uploaded_files)} documents.")

# Chat interface
if uploaded_files:
    query = st.text_input("Ask a question about your documents:")
    if query:
        query_vec = embedder.encode([query])
        D, I = index.search(query_vec, k=5)
        retrieved_docs = [all_chunks[i] for i in I[0]]

        context = "\n".join(retrieved_docs)
        result = rag_pipeline(question=query, context=context)

        st.subheader("💬 Answer")
        st.write(result['answer'])

        with st.expander("🔍 Retrieved Passages"):
            for passage in retrieved_docs:
                st.markdown(f"- {passage}")

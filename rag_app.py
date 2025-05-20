import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
import docx2txt
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load Hugging Face Token securely
HF_TOKEN = st.secrets["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"]
os.environ["HF_TOKEN"] = HF_TOKEN

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    rag_pipeline = pipeline(
        "text2text-generation", 
        model="google/flan-t5-base", 
        tokenizer="google/flan-t5-base",
        use_auth_token=HF_TOKEN
    )
    return embedder, rag_pipeline

embedder, rag_pipeline = load_models()

# File handling
def read_file(file):
    if file.type == "application/pdf":
        pdf = PdfReader(file)
        return "\n".join(page.extract_text() or '' for page in pdf.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

# Index creation
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Streamlit UI
st.title("📄 RAG App: Chat with Your Documents")

uploaded_files = st.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    full_text = ""
    for file in uploaded_files:
        full_text += read_file(file) + "\n"

    chunks = chunk_text(full_text)
    index, embeddings = create_faiss_index(chunks)

    st.success("✅ Documents processed and indexed!")

    query = st.text_input("Ask a question about your documents:")
    if query:
        query_embedding = embedder.encode([query])
        D, I = index.search(query_embedding, k=3)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = " ".join(relevant_chunks)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        with st.spinner("Generating answer..."):
            answer = rag_pipeline(prompt, max_length=200, do_sample=False)[0]["generated_text"]
        st.markdown("### 🧠 Answer")
        st.write(answer)

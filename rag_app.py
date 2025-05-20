import streamlit as st
import PyPDF2
import docx2txt
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="RAG App", layout="wide")

st.title("📄 RAG Chatbot: Ask Questions from Your Docs")

# Hugging Face Model
HF_TOKEN = st.secrets["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"]
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", token=HF_TOKEN)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

documents = []
for file in uploaded_files:
    text = extract_text(file)
    documents.append(text)

if documents:
    full_text = " ".join(documents)
    # Split into chunks
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    # Embedding & FAISS index
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.success("Documents loaded. Ask your question below.")

    query = st.text_input("Ask a question:")
    if query:
        query_embedding = embedder.encode([query])
        _, I = index.search(np.array(query_embedding), k=3)
        context = " ".join([chunks[i] for i in I[0]])

        result = qa_pipeline(question=query, context=context)
        st.subheader("Answer:")
        st.write(result["answer"])

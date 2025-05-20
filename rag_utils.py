import os
import fitz  # PyMuPDF
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)  # 384 dim for MiniLM
documents = []

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

def add_documents(files):
    global documents
    all_texts = []
    for file in files:
        text = extract_text(file)
        documents.append(text)
        all_texts.append(text)
    embeddings = model.encode(all_texts)
    index.add(np.array(embeddings))
    return "Documents embedded successfully."

def query_rag(question):
    q_embed = model.encode([question])
    D, I = index.search(np.array(q_embed), k=3)
    retrieved = [documents[i] for i in I[0]]
    context = "\n".join(retrieved)
    return f"Answer (context-based):\n{context}\n\n[Mock Answer Here – Connect to LLM]"

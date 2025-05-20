import os
import PyPDF2
import docx
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login

# Login to Hugging Face
login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def retrieve(query, index, chunks, embeddings, top_k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(context, question):
    return qa_pipeline(question=question, context=context)["answer"]

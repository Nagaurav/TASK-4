import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
documents = []

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

def add_documents(files):
    global documents
    texts = []
    for file in files:
        content = extract_text(file)
        if content:
            documents.append(content)
            texts.append(content)
    if texts:
        embeddings = model.encode(texts)
        index.add(np.array(embeddings))
    return "Documents embedded successfully."

def query_rag(question):
    q_embed = model.encode([question])
    D, I = index.search(np.array(q_embed), k=3)
    retrieved = [documents[i] for i in I[0] if i < len(documents)]
    context = "\n\n".join(retrieved)
    return f"Answer using top docs:\n\n{context}\n\n(Note: Mock answer here – no actual LLM response)"

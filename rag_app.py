import streamlit as st
from langchain.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import tempfile
import os

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📄 Retrieval-Augmented Generation (RAG) Chat")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.info("Processing documents...")
    raw_text = ""
    for file in uploaded_files:
        suffix = file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs = loader.load()
        raw_text += "\n".join([doc.page_content for doc in docs])
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    st.success("Documents ready. Ask your question below.")

    query = st.text_input("Ask something about the uploaded documents:")

    if query:
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        llm = OpenAI()
        response = llm(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        st.markdown("**Answer:**")
        st.write(response)

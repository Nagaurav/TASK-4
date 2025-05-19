import streamlit as st
from langchain.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import tempfile

# Set API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# UI
st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("📄 Chat with Your Documents using RAG")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
query = st.text_input("Ask a question about the uploaded documents:")

if uploaded_files and query:
    docs = []
    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {suffix}")
            continue

        docs.extend(loader.load())

    # Chunk & Embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # RetrievalQA
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)

    # Response
    response = qa.run(query)
    st.markdown("### 📌 Answer:")
    st.success(response)

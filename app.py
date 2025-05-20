import streamlit as st
from rag_utils import read_txt, read_pdf, read_docx, chunk_text, create_faiss_index, retrieve, generate_answer

st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("🧠 RAG Chat with Your Documents")

uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.embeddings = None
    st.session_state.chunks = []

if uploaded_files:
    raw_text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            raw_text += read_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw_text += read_docx(file)
        elif file.type == "text/plain":
            raw_text += read_txt(file)

    chunks = chunk_text(raw_text)
    index, embeddings, stored_chunks = create_faiss_index(chunks)

    st.session_state.index = index
    st.session_state.embeddings = embeddings
    st.session_state.chunks = stored_chunks

    st.success(f"{len(chunks)} chunks indexed. You can now ask questions.")

if st.session_state.index:
    question = st.text_input("Ask a question about the documents:")
    if question:
        context_passages = retrieve(
            query=question,
            index=st.session_state.index,
            chunks=st.session_state.chunks,
            embeddings=st.session_state.embeddings,
        )
        context = " ".join(context_passages)
        answer = generate_answer(context, question)

        st.subheader("📄 Context used:")
        st.write(context)

        st.subheader("🧠 Answer:")
        st.success(answer)

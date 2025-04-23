import os
import pickle
import streamlit as st
import faiss
import docx2txt
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime
from functools import lru_cache

# 1. Gemini API Key
genai.configure(api_key="AIzaSyDdDhqFxX3a26S7GogaC6H3bFBBg6UupvY")  # Replace securely with st.secrets or env var in production

# 2. Load documents (.pdf, .docx)
def load_document(file_path):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return [doc for doc in docs if doc.page_content.strip()]
        elif file_path.endswith(".docx"):
            text = docx2txt.process(file_path)
            return [Document(page_content=text)]
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
    return []

# 3. Chunking
def chunk_documents(docs, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# 4. Embeddings & Indexing
@lru_cache(maxsize=None)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(text_chunks, embedder, index_path, chunk_path):
    if os.path.exists(index_path) and os.path.exists(chunk_path):
        return
    embeddings = embedder.encode(text_chunks).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(chunk_path, "wb") as f:
        pickle.dump(text_chunks, f)

def get_relevant_chunks(query, embedder, index_path, chunk_path, k=3):
    index = faiss.read_index(index_path)
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    query_embedding = embedder.encode([query]).astype('float32')
    _, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

 
# 6. Answer Query
def answer_with_context(question, embedder):
    # Read all files in data/ regardless of mapping
    document_paths = [
        os.path.join("data", f)
        for f in os.listdir("data")
        if f.endswith(('.pdf', '.docx'))
    ]

    all_chunks = []
    opened_files = []


    for path in document_paths:
        docs = load_document(path)
        if docs:
            chunks = chunk_documents(docs)
            all_chunks.extend([chunk.page_content for chunk in chunks])
            
            opened_files.append(os.path.basename(path))

    if not all_chunks:
        return "⚠️ No valid content found.", []

    index_path = "index.faiss"
    chunk_path = "chunks.pkl"
    create_vector_store(all_chunks, embedder, index_path, chunk_path)

    relevant_chunks = get_relevant_chunks(question, embedder, index_path, chunk_path)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful assistant for ASBL Real Estate.
Use the provided context to answer the user's question. If the answer isn’t directly available, try to infer or provide a helpful explanation using relevant details from the context. If truly unavailable, say so politely.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip(), opened_files
    except Exception as e:
        return f"❌ Gemini error: {e}", opened_files

# 7. Streamlit UI
st.set_page_config(page_title="ASBL CRM Bot", page_icon="🏢", layout="wide")

st.markdown("<h1 style='text-align: center; color: #003566;'>👨🏻‍💼 Hi! I'm Dhruv.</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Your personal AI assistant for all your purchase queries at ASBL.</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📂 Document Info")
    os.makedirs("data", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith(('.pdf', '.docx'))]
    if files:
        st.success(f"{len(files)} document(s) loaded.")
        for file in files:
            st.markdown(f"- 📄 {file}")
    else:
        st.warning("No documents found in `data/` folder.")

# Embedder init
embedder = get_embedder()

# Chat logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask a real estate question...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("🔍 Thinking..."):
        answer, opened = answer_with_context(question, embedder)
    st.session_state.chat_history.append({"role": "bot", "content": answer})

    st.markdown("### 🤖 Answer:")
    st.markdown(answer)

    if opened:
        st.markdown("### ✅ Referenced Document(s):")
        for doc in opened:
            st.markdown(f"- 📄 {doc}")

# Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Download Option
if st.session_state.chat_history:
    with st.expander("💾 Download Chat History"):
        chat_log = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history)
        fname = f"asbl_faq_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button("📥 Download", chat_log, file_name=fname)

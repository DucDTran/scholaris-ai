import os
import shutil
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.load_vector_store import get_embeddings_model

def process_and_save_document(text_chunks, doc_store_path):
    """Creates a FAISS vector store and saves it."""
    if not text_chunks:
        st.error("Could not split document into chunks.")
        return False
    
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    if os.path.exists(doc_store_path):
        shutil.rmtree(doc_store_path)  # Remove old version if re-processing
    vector_store.save_local(doc_store_path)
    return True
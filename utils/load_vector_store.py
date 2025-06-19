import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vector_store(user_path, doc_name):
    """Loads a single FAISS vector store from the local file system."""
    embeddings = get_embeddings_model()
    vector_store_path = os.path.join(user_path, "vector_stores", doc_name)
    if os.path.exists(vector_store_path):
        try:
            return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load document '{doc_name}': {e}")
            return None
    else:
        st.error(f"Could not find processed document: {doc_name}")
        return None
    

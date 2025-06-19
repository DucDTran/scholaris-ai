import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_and_merge_vector_stores(user_path, doc_names):
    """Loads multiple FAISS vector stores and merges them into one."""
    embeddings = get_embeddings_model()
    main_vector_store = None
    vector_stores_path = os.path.join(user_path, "vector_stores")

    for i, doc_name in enumerate(doc_names):
        doc_path = os.path.join(vector_stores_path, doc_name)
        if not os.path.exists(doc_path):
            st.warning(f"Could not find processed document: {doc_name}")
            continue
        
        try:
            if i == 0:
                main_vector_store = FAISS.load_local(doc_path, embeddings, allow_dangerous_deserialization=True)
            else:
                new_vector_store = FAISS.load_local(doc_path, embeddings, allow_dangerous_deserialization=True)
                main_vector_store.merge_from(new_vector_store)
        except Exception as e:
            st.error(f"Error loading document '{doc_name}': {e}")
            
    return main_vector_store
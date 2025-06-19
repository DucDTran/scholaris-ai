import os
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st

def get_text_from_file(uploaded_file):
    """Extracts text from a single uploaded file object."""
    text = ""
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    if file_extension == ".pdf":
        try:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {file_name}: {e}")
    elif file_extension == ".docx":
        try:
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.error(f"Error reading {file_name}: {e}")
    elif file_extension == ".txt":
        try:
            text += uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading {file_name}: {e}")
    return text
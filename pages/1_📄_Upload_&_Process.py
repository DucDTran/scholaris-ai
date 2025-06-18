
import streamlit as st
import os
import re
import shutil
from PyPDF2 import PdfReader
from docx import Document
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import arxiv

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Upload & Process", page_icon="📄", layout="wide")

# --- FUNCTIONS ---
def sanitize_filename(filename):
    """Removes special characters to create a valid directory name."""
    s = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
    return s[:100]  # Truncate to a reasonable length

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

def get_text_from_pdf_path(pdf_path):
    """Extracts text from a PDF file path using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(raw_text)

def get_processed_documents(user_path):
    """Returns a list of already processed document names."""
    vector_stores_path = os.path.join(user_path, "vector_stores")
    if os.path.exists(vector_stores_path):
        return [name for name in os.listdir(vector_stores_path) if os.path.isdir(os.path.join(vector_stores_path, name))]
    return []

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

# --- UI & LOGIC ---
st.title("📄 Upload and Process Your Documents")
st.markdown("Upload your documents to create a library of exercises. You can also search for papers and process them.")

if "user_name" not in st.session_state or not st.session_state.get("user_name"):
    st.warning("Please login to continue.", icon="⚠️")
    st.stop()

if st.session_state.get("google_api_key") is None:
    st.warning("Please enter your Google API Key to continue.", icon="⚠️")
    st.stop()

user_name = st.session_state.user_name
google_api_key = st.session_state.google_api_key
user_id = st.session_state.user_id
user_data_path = os.path.join("user_data", user_id)

# Add user info and logout button to the sidebar
st.sidebar.write(f"Welcome, **{user_name}**!")

vector_stores_path = os.path.join(user_data_path, "vector_stores")
if not os.path.exists(vector_stores_path):
    os.makedirs(vector_stores_path)

st.subheader(f"Document Management for: {user_name}")

# --- TABS FOR UPLOAD AND ARXIV SEARCH ---
tab1, tab2 = st.tabs(["Upload Your Files", "Search Arxiv for Papers"])

with tab1:
    st.header("Upload from Your Computer")
    uploaded_files = st.file_uploader(
        "Choose your documents to upload (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt']
    )
    force_reprocess_upload = st.checkbox("Re-process uploaded files if they already exist.")

    if st.button("Process Uploaded Files", type="primary"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                sanitized_name = sanitize_filename(os.path.splitext(uploaded_file.name)[0])
                doc_store_path = os.path.join(vector_stores_path, sanitized_name)
                
                if os.path.exists(doc_store_path) and not force_reprocess_upload:
                    st.write(f"↪️ Skipping '{uploaded_file.name}', already processed.")
                    continue

                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    raw_text = get_text_from_file(uploaded_file)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        if process_and_save_document(text_chunks, doc_store_path):
                            st.write(f"✅ Successfully processed '{uploaded_file.name}'.")
                    else:
                        st.error(f"Could not extract text from '{uploaded_file.name}'.")
            st.success("File processing complete!")
        else:
            st.warning("Please upload at least one file.")

with tab2:
    st.header("Find and Process Papers")
    arxiv_query = st.text_input("Enter a topic to search", "large language models")

    if st.button("Search", type="primary"):
        with st.spinner("Searching..."):
            try:
                # arxiv_results = arxiv_client.load(query=arxiv_query)
                arxiv_search = arxiv.Search(arxiv_query, max_results=1)
                arxiv_results = list(arxiv_search.results())
                st.session_state.all_results = arxiv_results
            except Exception as e:
                st.error(f"Failed to search papers: {e}")
                st.session_state.all_results = []

    selected_papers_for_processing = []
    if "all_results" in st.session_state and st.session_state.all_results:
        st.markdown("---")

        with st.form(key="paper_form"):
            st.subheader("Top Results")        
            for idx, paper in enumerate(st.session_state.all_results):
                paper_id = idx
                paper_title = paper.title
                paper_link = paper.entry_id
                paper_authors = paper.authors
                paper_summary = paper.summary

                with st.container():
                    col1, col2 = st.columns([0.2, 0.8])
                    with col1:
                        if st.checkbox("Add to Library", key=f"paper_{paper_id}"):
                            selected_papers_for_processing.append(paper)
                            st.write(f"✅ Successfully added '{paper_title}' to the library.")
                    with col2:
                        st.markdown(f"#### {paper_title}")
                        st.markdown(f"**Link:**\n{paper_link}")
                        st.markdown(f"**Author(s):** {paper_authors}")
                        st.markdown(f"**Summary:**\n{paper_summary}")
                        
                    st.markdown("---")
            submitted = st.form_submit_button("Process Selected Papers", type="primary")
    
        st.write(f"Selected {len(selected_papers_for_processing)} papers for processing.")

        if submitted:
            if len(selected_papers_for_processing) == 0:
                st.warning("Please select at least one paper to process.")
            else:
                with st.spinner("Downloading and processing selected papers..."):
                    for paper in selected_papers_for_processing:
                        sanitized_name = sanitize_filename(paper.title)
                        doc_store_path = os.path.join(vector_stores_path, sanitized_name)
                        try:
                            pdf_path = paper.download_pdf(dirpath=vector_stores_path)
                            raw_text = get_text_from_pdf_path(pdf_path)
                            if raw_text.strip():
                                text_chunks = get_text_chunks(raw_text)
                                if process_and_save_document(text_chunks, doc_store_path):
                                    st.write(f"✅ Successfully processed '{paper.title}'.")
                            else:
                                st.error(f"Could not extract text from '{paper.title}'.")
                        except Exception as e:
                            st.error(f"Failed to process '{paper.title}': {e}")
                st.success("Paper processing complete!")
                # Clear results after processing to avoid reprocessing
                del st.session_state.all_results


# --- Display Processed Documents Library ---
st.markdown("---")
st.subheader("Your Processed Document Library")
processed_docs = get_processed_documents(user_data_path)

if not processed_docs:
    st.info("You haven't processed any documents yet. Upload some files or search papers to get started.")
else:
    st.write("The following documents are in your library and available for exercises:")
    for doc_name in processed_docs:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f" ✅ `{doc_name}`")
        with col2:
            if st.button("Remove", key=f"remove_{doc_name}"):
                doc_to_delete_path = os.path.join(vector_stores_path, doc_name)
                try:
                    shutil.rmtree(doc_to_delete_path)
                    st.success(f"Removed '{doc_name}' successfully.")
                    st.rerun() # Refresh the page to update the list
                except Exception as e:
                    st.error(f"Failed to remove '{doc_name}': {e}")

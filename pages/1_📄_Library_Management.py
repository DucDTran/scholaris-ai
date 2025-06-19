
import streamlit as st
import os
import shutil
import arxiv
from utils.sanitize_filename import sanitize_filename
from utils.get_text_from_file import get_text_from_file
from utils.get_text_from_pdf_path import get_text_from_pdf_path
from utils.get_text_chunks import get_text_chunks
from utils.get_processed_documents import get_processed_documents
from utils.process_and_save_documents import process_and_save_document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Upload & Process", page_icon="📄", layout="wide")

# --- UI & LOGIC ---
st.title("📄 Upload or Search Documents")
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
st.sidebar.button("Logout", on_click=st.logout)

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
                client = arxiv.Client()
                arxiv_search = arxiv.Search(arxiv_query, max_results=4)
                arxiv_results = list(client.results(arxiv_search))
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
                paper_authors = ", ".join([author.name for author in paper.authors])
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

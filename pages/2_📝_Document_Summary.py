import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from utils.get_processed_documents import get_processed_documents
from utils.load_vector_store import load_vector_store


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Summarization", page_icon="📝", layout="wide")

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.2)

def initialize_summary_state():
    st.session_state.generated_summary = None
    st.session_state.summarized_doc_name = None
    st.session_state.summary_saved = False
    st.session_state.summary_displayed = False

# --- UI & LOGIC ---
st.title("📝 Document Summarization")
st.markdown("Select one of your processed documents to generate a summary.")

if "user_name" not in st.session_state or not st.session_state.get("user_name"):
    st.warning("Please login to continue.", icon="⚠️")
    st.stop()

if st.session_state.get("google_api_key") is None:
    st.warning("Please enter your Google API Key to continue.", icon="⚠️")
    st.stop()

# Get user info and set up paths
user_name = st.session_state.user_name
google_api_key = st.session_state.google_api_key
user_id = st.session_state.user_id
user_data_path = os.path.join("user_data", user_id)
llm = get_llm(google_api_key)

st.sidebar.write(f"Welcome, **{user_name}**!")
st.sidebar.button("Logout", on_click=st.logout)

if 'generated_summary' not in st.session_state:
    initialize_summary_state()
    # --- Summarization Workflow ---
st.subheader("Summarization Setup")

processed_docs = get_processed_documents(user_data_path)
if not processed_docs:
    st.warning("You have no processed documents. Please go to 'Library Management' to add some.", icon="⚠️")
    st.stop()

selected_doc = st.selectbox(
    "Choose a document to summarize:",
    options=processed_docs,
    index=None,
    placeholder="Select a document..."
)

if st.button("Generate Summary", type="primary"):
    if 'generated_summary' in st.session_state:
        del st.session_state.generated_summary
    if 'summarized_doc_name' in st.session_state:
        del st.session_state.summarized_doc_name
    if 'summary_saved' in st.session_state:
        del st.session_state.summary_saved
    if 'summary_displayed' in st.session_state:
        del st.session_state.summary_displayed

    if not selected_doc:
        st.error("Please select a document to summarize.")
    else:
        with st.spinner(f"Loading '{selected_doc}' and generating summary..."):
            vector_store = load_vector_store(user_data_path, selected_doc)
            if vector_store:
                try:

                    all_docs = vector_store.similarity_search("", k=999)

                    if not all_docs:
                        st.error("The selected document appears to be empty.")
                    else:
                        prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant designed to create a summary of the provided context. The summary should be comprehensive and to the point, focusing on the main topics of the context. The summary should be in a markdown format, with headers, subheaders and bullet points whenever appropriate."),
                        ("human", "Context:\n---\n{context}")
                    ])
                        chain = prompt | llm
                        summary = chain.invoke({"context": all_docs})

                        st.session_state.generated_summary = summary.content
                        st.session_state.summarized_doc_name = selected_doc
                        st.session_state.summary_displayed = True

                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")

if st.session_state.get("summary_displayed") or st.session_state.get("summary_saved"):
    st.markdown("---")
    st.subheader("Generated Summary")
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(st.session_state.generated_summary)
    with col2:
        if st.button("Save Summary to Library", key="save_summary"):
            with st.spinner("Adding summary to the document's vector store..."):
                summary_to_save = st.session_state.generated_summary
                doc_name_to_update = st.session_state.summarized_doc_name
                
                # Load the vector store again to modify it
                vector_store_to_update = load_vector_store(user_data_path, doc_name_to_update)
                if vector_store_to_update:
                    try:
                        # Add the new summary text to the existing in-memory index
                        vector_store_to_update.add_texts([summary_to_save])
                        
                        # Get the path to save the updated index
                        vector_store_path = os.path.join(user_data_path, "vector_stores", doc_name_to_update)
                        
                        # Save the modified index, overwriting the old one
                        vector_store_to_update.save_local(vector_store_path)
                        
                        st.success(f"Summary successfully added to the '{doc_name_to_update}' library item.")
                        st.session_state.summary_saved = True

                    except Exception as e:
                        st.error(f"Failed to save the summary: {e}")
                else:
                    st.error("Could not load the document to save the summary. Please try again.")

        if st.download_button("Export Summary to Markdown", 
                              key="export_summary", 
                              data=st.session_state.generated_summary, 
                              file_name=f"{st.session_state.summarized_doc_name}_summary.md"):
            st.write("Summary exported to Markdown.")   

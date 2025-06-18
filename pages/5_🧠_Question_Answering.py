import streamlit as st
import os
import json
import random
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np
from scipy.spatial.distance import cosine

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Generated Q&A",
    page_icon="🧠",
    layout="wide"
)

# --- Pydantic Data Structures ---
class GeneratedQuestion(BaseModel):
    """Pydantic model for a generated question based on a document chunk."""
    question: str = Field(description="A challenging, open-ended question based on the provided context.")
    ideal_answer: str = Field(description="A detailed, ideal answer to the question, synthesized from the context.")

# --- FUNCTIONS ---
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.4)

def get_processed_documents(user_path):
    vector_stores_path = os.path.join(user_path, "vector_stores")
    if os.path.exists(vector_stores_path):
        return [name for name in os.listdir(vector_stores_path) if os.path.isdir(os.path.join(vector_stores_path, name))]
    return []

def load_and_merge_vector_stores(user_path, doc_names):
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

def save_average_similarity_score(user_path, average_score):
    scores_file = os.path.join(user_path, "scores.json")
    try:
        scores_data = []
        if os.path.exists(scores_file):
            with open(scores_file, "r") as f:
                scores_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        scores_data = []
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "Q&A Similarity",
        "score": average_score,
        "total": 100,
        "percentage": average_score
    }
    scores_data.append(new_entry)
    with open(scores_file, "w") as f:
        json.dump(scores_data, f, indent=4)

def generate_qa_exercises(vector_store, llm, num_questions):
    all_docs = vector_store.similarity_search("", k=100)
    if not all_docs:
        st.error("Could not retrieve any content from the selected documents.")
        return []
    
    num_to_generate = min(len(all_docs), num_questions)
    if len(all_docs) < num_questions:
        st.warning(f"Not enough content to generate {num_questions} questions. Generating {num_to_generate} instead.")

    selected_docs = random.sample(all_docs, num_to_generate)
    
    parser = JsonOutputParser(pydantic_object=GeneratedQuestion)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant designed to create question and answer exercises based ONLY on the provided context. Each exercise should have a question and a short, concise ideal answer. The answer should be given under bullet points whenever possible. Format as a JSON object."),
        ("human", "Context:\n---\n{context}\n---\n\nFormat Instructions:\n{format_instructions}")
    ])
    chain = prompt | llm | parser

    exercises = []
    progress_bar = st.progress(0, text="Generating exercises...")
    for i, doc in enumerate(selected_docs):
        try:
            response = chain.invoke({"context": doc.page_content, "format_instructions": parser.get_format_instructions()})
            if isinstance(response, list) and response:
                response = response[0]
            if isinstance(response, dict):
                response['source_context'] = doc.page_content
                exercises.append(response)
        except Exception as e:
            st.error(f"Error generating question {i+1}: {e}")
        progress_bar.progress((i + 1) / len(selected_docs))
    return exercises

def calculate_similarity(embedding_model, text1, text2):
    if not text1 or not text2:
        return 0
    # FIX: Ensure vectors are 1-D by not reshaping them.
    # The `embed_query` method already returns a 1-D list.
    vec1 = np.array(embedding_model.embed_query(text1))
    vec2 = np.array(embedding_model.embed_query(text2))
    similarity = 1 - cosine(vec1, vec2)
    return round(similarity * 100)

def initialize_qa_state():
    st.session_state.qa_exercises = []
    st.session_state.qa_current_index = 0
    st.session_state.qa_user_answers = {}
    st.session_state.qa_similarity_scores = {}
    st.session_state.qa_started = False
    st.session_state.qa_submitted = False
    st.session_state.selected_docs_for_qa = []

# --- UI & LOGIC ---
st.title("🧠 Generated Question & Answer")
st.markdown("Select the documents you want to be quizzed on. Each document will be processed individually so you can select them for exercises.")

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
llm = get_llm(google_api_key)
embedding_model = get_embeddings_model()

st.sidebar.write(f"Welcome, **{user_name}**!")
st.sidebar.button("Logout", on_click=st.logout)

processed_docs = get_processed_documents(user_data_path)
if not processed_docs:
    st.warning("You have no processed documents. Please go to 'Upload & Process' to add some.", icon="⚠️")
    st.stop()

if 'qa_started' not in st.session_state:
    initialize_qa_state()

# --- Results Page ---
if st.session_state.get('qa_submitted'):
    st.header("Similarity Results")
    scores = list(st.session_state.qa_similarity_scores.values())
    average_score = round(np.mean(scores)) if scores else 0
    
    st.metric("Average Similarity Score", f"{average_score}%")
    save_average_similarity_score(user_data_path, average_score)

    for i, exercise in enumerate(st.session_state.qa_exercises):
        st.markdown("---")
        st.markdown(f"Question {i+1}: {exercise['question']}")
        
        score = st.session_state.qa_similarity_scores.get(i, 0)
        st.info(f"Similarity Score: {score}%")

        col1, col2 = st.columns(2)
        with col1:
            st.success("Your Answer")
            st.write(st.session_state.qa_user_answers.get(i, "*No answer provided*"))
        with col2:
            st.warning("Ideal Answer from Source")
            with st.expander("Show Ideal Answer"):
                st.write(exercise['ideal_answer'])

        with st.expander("Show Original Source Text"):
            st.markdown(f"> {exercise['source_context']}")

    if st.button("Try Another Set"):
        initialize_qa_state()
        st.rerun()

# --- Gameplay Page ---
elif st.session_state.get('qa_started'):
    st.info(f"Exercise based on: {', '.join(st.session_state.selected_docs_for_qa)}")
    current_index = st.session_state.qa_current_index
    exercise = st.session_state.qa_exercises[current_index]

    st.subheader(f"Question {current_index + 1} of {len(st.session_state.qa_exercises)}")
    st.markdown(f"{exercise['question']}")

    st.session_state.qa_user_answers[current_index] = st.text_area(
        "Your Answer:",
        value=st.session_state.qa_user_answers.get(current_index, ""),
        key=f"qa_answer_{current_index}",
        height=150
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if current_index > 0:
            if st.button("⬅️ Previous"):
                st.session_state.qa_current_index -= 1
                st.rerun()
    with col2:
        if current_index < len(st.session_state.qa_exercises) - 1:
            if st.button("Next ➡️"):
                st.session_state.qa_current_index += 1
                st.rerun()
    with col3:
        if st.button("Submit All Answers", type="primary"):
            with st.spinner("Calculating all similarity scores..."):
                for i, ex in enumerate(st.session_state.qa_exercises):
                    user_ans = st.session_state.qa_user_answers.get(i, "")
                    ideal_ans = ex['ideal_answer']
                    st.session_state.qa_similarity_scores[i] = calculate_similarity(embedding_model, user_ans, ideal_ans)
            st.session_state.qa_submitted = True
            st.rerun()
    with col4:
        if st.button("Try Another Set", type="secondary"):
            initialize_qa_state()
            st.rerun()
# --- Setup Page ---
else:
    selected_docs = st.multiselect(
        "Select documents to generate questions from:",
        options=processed_docs,
        default=st.session_state.get('selected_docs_for_qa', [])
    )
    num_questions = st.number_input("How many questions to generate?", min_value=1, max_value=10, value=4)
    
    if st.button("Generate Exercises", type="primary"):
        if not selected_docs:
            st.error("Please select at least one document.")
        else:
            initialize_qa_state()
            st.session_state.selected_docs_for_qa = selected_docs
            with st.spinner("Loading documents and generating exercises..."):
                vector_store = load_and_merge_vector_stores(user_data_path, selected_docs)
                if vector_store:
                    exercises = generate_qa_exercises(vector_store, llm, num_questions)
                    if exercises:
                        st.session_state.qa_exercises = exercises
                        st.session_state.qa_started = True
                        st.rerun()
                    else:
                        st.error("Could not generate exercises from the selected documents.")
                else:
                    st.error("Could not load the selected documents.")

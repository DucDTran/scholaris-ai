import streamlit as st
import os
import random
import json
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multiple Choice Quiz", page_icon="❓", layout="wide")

# --- Pydantic Data Structure ---
class MCQuestion(BaseModel):
    question: str = Field(description="The multiple-choice question.")
    options: List[str] = Field(description="A list of 4 options for the question. The first option must be the correct one.")
    explanation: str = Field(description="A brief explanation of why the correct answer is correct, based on the context.")

# --- FUNCTIONS ---
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.3)

def get_processed_documents(user_path):
    vector_stores_path = os.path.join(user_path, "vector_stores")
    if os.path.exists(vector_stores_path):
        return [name for name in os.listdir(vector_stores_path) if os.path.isdir(os.path.join(vector_stores_path, name))]
    return []

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

def save_user_score(user_path, score, total):
    # ... (function remains unchanged)
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
        "type": "Multiple Choice Quiz",
        "score": score,
        "total": total,
        "percentage": round((score / total) * 100, 2) if total > 0 else 0
    }
    scores_data.append(new_entry)
    with open(scores_file, "w") as f:
        json.dump(scores_data, f, indent=4)


def generate_quiz_questions(vector_store, llm, num_questions):
    all_docs = vector_store.similarity_search("", k=100)
    if not all_docs:
        st.error("Could not retrieve any documents from the selected vector store(s).")
        return []
    if len(all_docs) < num_questions:
        st.warning(f"Not enough content in selected docs to generate {num_questions} questions. Generating {len(all_docs)} instead.")
        num_questions = len(all_docs)
    
    selected_docs = random.sample(all_docs, num_questions)
    
    parser = JsonOutputParser(pydantic_object=MCQuestion)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant designed to create multiple-choice quizzes based ONLY on the provided context. Each question should have 4 options, with one correct answer. The questions should test key concepts from the text, focusing on main topics of the context. Avoid asking questions that are too similar to each other. Avoid asking trivial questions. Avoid asking questions related to the page numbers or the document name."),
        ("human", "Context:\n---\n{context}\n---\n\nFormat Instructions:\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    questions = []
    progress_bar = st.progress(0, text="Generating questions...")
    for i, doc in enumerate(selected_docs):
        try:
            response = chain.invoke({"context": doc.page_content, "format_instructions": parser.get_format_instructions()})
            
            # FIX: Handle cases where the LLM wraps the response in a list.
            if isinstance(response, list) and response:
                response = response[0]

            if isinstance(response, dict):
                response['source'] = doc.page_content
                questions.append(response)
            else:
                st.warning(f"LLM returned a malformed response (type: {type(response)}). Skipping this question.")

        except Exception as e:
            # This will catch other errors during the invocation itself.
            st.error(f"An error occurred while generating question {i+1}: {e}. Skipping.")
        progress_bar.progress((i + 1) / num_questions)
    return questions


def initialize_quiz_state():
    st.session_state.questions = []
    st.session_state.current_q_index = 0
    st.session_state.user_answers = {}
    st.session_state.quiz_started = False
    st.session_state.quiz_submitted = False
    st.session_state.selected_docs_for_quiz = []

def record_answer():
    q_index = st.session_state.current_q_index
    answer = st.session_state[f"radio_{q_index}"]
    st.session_state.user_answers[q_index] = answer

# --- UI & LOGIC ---
st.title("❓ Multiple Choice Quiz")
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

st.sidebar.write(f"Welcome, **{user_name}**!")
st.sidebar.button("Logout", on_click=st.logout)

processed_docs = get_processed_documents(user_data_path)
if not processed_docs:
    st.warning("You have no processed documents. Please go to 'Upload & Process' to add some.", icon="⚠️")
    st.stop()

if 'quiz_started' not in st.session_state:
    initialize_quiz_state()

# --- Display Results Page ---
if st.session_state.get('quiz_submitted'):
    st.header("Quiz Results")
    score = 0
    questions = st.session_state.questions
    user_answers = st.session_state.user_answers

    for i, q_data in enumerate(questions):
        user_choice_idx = user_answers.get(i)
        correct_idx = q_data['correct_answer_shuffled_index']
        is_correct = (user_choice_idx is not None and user_choice_idx == correct_idx)
        if is_correct:
            score += 1
        
        st.markdown("---")
        st.subheader(f"Question {i+1}: {q_data['question']}")
        
        user_answer_text = f"Your answer: **{q_data['options'][user_choice_idx]}**" if user_choice_idx is not None else "Your answer: *Not Answered*"
        correct_answer_text = f"Correct answer: **{q_data['options'][correct_idx]}**"
        
        if is_correct:
            st.success(f"{user_answer_text} ✔️")
        else:
            st.error(f"{user_answer_text} ❌")
            st.info(correct_answer_text)

        with st.expander("💡 Explanation & Source Document"):
            st.markdown(f"**Explanation:** {q_data['explanation']}")
            st.markdown(f"**Source:**\n\n> {q_data['source']}")
    
    st.markdown("---")
    final_score_text = f"Your final score: {score} / {len(questions)}"
    st.header(final_score_text)
    
    if st.button("Take Another Quiz"):
        initialize_quiz_state()
        st.rerun()
    st.stop()

# --- Quiz Setup & Gameplay ---
if not st.session_state.get('quiz_started'):
    selected_docs = st.multiselect(
        "Select the documents you want to be quizzed on:",
        options=processed_docs,
        default=st.session_state.get('selected_docs_for_quiz', [])
    )
    num_questions_to_gen = st.number_input("How many questions would you like?", min_value=1, max_value=10, value=4)

    if st.button("Generate Quiz", type="primary"):
        if not selected_docs:
            st.error("Please select at least one document.")
        else:
            initialize_quiz_state()
            st.session_state.selected_docs_for_quiz = selected_docs
            with st.spinner("Generating Questions..."):
                vector_store = load_and_merge_vector_stores(user_data_path, selected_docs)
                if vector_store:
                    questions = generate_quiz_questions(vector_store, llm, num_questions_to_gen)
                    if questions:
                        st.session_state.questions = questions
                        st.session_state.quiz_started = True
                        for q in st.session_state.questions:
                            correct_answer = q['options'][0]
                            random.shuffle(q['options'])
                            q['correct_answer_shuffled_index'] = q['options'].index(correct_answer)
                        st.rerun()
                    else:
                        st.error("Failed to generate questions from the selected documents.")
                else:
                    st.error("Could not load the selected documents. Please try again.")

else: # This block runs when quiz_started is True
    # ... (The quiz gameplay logic remains unchanged)
    st.markdown("---")
    st.info(f"Quiz based on: {', '.join(st.session_state.selected_docs_for_quiz)}")
    current_index = st.session_state.current_q_index
    q_data = st.session_state.questions[current_index]
    
    st.subheader(f"Question {current_index + 1} of {len(st.session_state.questions)}")
    st.markdown(f"#### {q_data['question']}")

    st.radio(
        "Choose your answer:",
        options=range(len(q_data['options'])),
        format_func=lambda i: q_data['options'][i],
        key=f"radio_{current_index}",
        index=st.session_state.user_answers.get(current_index),
        on_change=record_answer,
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.session_state.current_q_index > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_q_index -= 1
                st.rerun()
    with col2:
        if st.session_state.current_q_index < len(st.session_state.questions) - 1:
            if st.button("Next ➡️"):
                st.session_state.current_q_index += 1
                st.rerun()
    with col3:
        if st.button("Submit Quiz", type="primary"):
            st.session_state.quiz_submitted = True
            final_score = 0
            for i, q in enumerate(st.session_state.questions):
                if st.session_state.user_answers.get(i) == q['correct_answer_shuffled_index']:
                    final_score += 1
            save_user_score(user_data_path, final_score, len(st.session_state.questions))
            st.rerun()
    with col4:
        if st.button("Try Another Quiz", type="secondary"):
            initialize_quiz_state()
            st.rerun()
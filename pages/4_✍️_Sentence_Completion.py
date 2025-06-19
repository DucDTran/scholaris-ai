import streamlit as st
import os
import random
import json
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from utils.load_and_merge_vector_store import load_and_merge_vector_stores
from utils.get_processed_documents import get_processed_documents
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sentence Completion", page_icon="✍️", layout="wide")

# --- Pydantic Data Structure ---
class FillInTheBlank(BaseModel):
    sentence_with_blank: str = Field(description="A sentence from the text with a key term replaced by '[BLANK]'.")
    correct_answer: str = Field(description="The key term that was removed from the sentence.")
    explanation: str = Field(description="A brief explanation of the key term or why it's important in the context.")

# --- FUNCTIONS ---

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.3)

def save_user_score(user_path, score, total):
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
        "type": "Sentence Completion",
        "score": score,
        "total": total,
        "percentage": round((score / total) * 100, 2) if total > 0 else 0
    }
    scores_data.append(new_entry)
    with open(scores_file, "w") as f:
        json.dump(scores_data, f, indent=4)

def generate_fill_in_the_blank_exercises(vector_store, llm, num_exercises):
    all_docs = vector_store.similarity_search("", k=100)
    if not all_docs:
        st.error("Could not retrieve any content from the selected documents.")
        return []
    if len(all_docs) < num_exercises:
        st.warning(f"Not enough content to generate {num_exercises} exercises. Generating {len(all_docs)} instead.")
        num_exercises = len(all_docs)
    
    selected_docs = random.sample(all_docs, num_exercises)
    
    parser = JsonOutputParser(pydantic_object=FillInTheBlank)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant designed to create fill-in-the-blank exercises based ONLY on the provided context. Each exercise should have a key sentence with a blank, and the correct answer should be one of the key terms in the sentence. The explanation should be a brief explanation of the key term or why it's important in the context. Format as a JSON object."),
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
                response['source'] = doc.page_content
                exercises.append(response)
            else:
                st.warning(f"LLM returned a malformed response (type: {type(response)}). Skipping this exercise.")

        except Exception as e:
            st.error(f"An error occurred while generating exercise {i+1}: {e}. Skipping.")
        progress_bar.progress((i + 1) / num_exercises)

    return exercises

def initialize_sc_state():
    st.session_state.sc_exercises = []
    st.session_state.sc_current_index = 0
    st.session_state.sc_score = 0
    st.session_state.sc_user_answers = {}
    st.session_state.sc_started = False
    st.session_state.sc_submitted = False
    st.session_state.selected_docs_for_sc = []

# --- UI & LOGIC ---
st.title("✍️ Sentence Completion")
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

if 'sc_started' not in st.session_state:
    initialize_sc_state()

# --- Display Results Page ---
if st.session_state.get('sc_submitted'):
    st.header("Exercise Results")
    score = 0
    exercises = st.session_state.sc_exercises
    user_answers = st.session_state.sc_user_answers

    for i, ex_data in enumerate(exercises):
        user_answer = user_answers.get(i, "")
        correct_answer = ex_data['correct_answer']
        is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
        if is_correct:
            score += 1
        
        st.markdown("---")
        full_sentence = ex_data['sentence_with_blank'].replace("[BLANK]", f"**{correct_answer}**")
        st.subheader(f"Exercise {i+1}: {full_sentence}")

        user_answer_text = f"Your answer: **{user_answer}**" if user_answer else "Your answer: *Not Answered*"
        
        if is_correct:
            st.success(f"{user_answer_text} ✔️")
        else:
            st.error(f"{user_answer_text} ❌")
            st.info(f"Correct answer: **{correct_answer}**")

        with st.expander("💡 Explanation & Source Document"):
            st.markdown(f"**Explanation:** {ex_data['explanation']}")
            st.markdown(f"**Source:**\n\n> {ex_data['source']}")
    
    st.markdown("---")
    st.header(f"Your final score: {score} / {len(exercises)}")
    if st.button("Try Another Set"):
        initialize_sc_state()
        st.rerun()
    st.stop()

# --- Exercise Setup & Gameplay ---
if not st.session_state.get('sc_started'):
    selected_docs = st.multiselect(
        "Select the documents for the exercise:",
        options=processed_docs,
        default=st.session_state.get('selected_docs_for_sc', [])
    )
    num_exercises_to_gen = st.number_input("How many exercises would you like?", min_value=1, max_value=10, value=4)

    if st.button("Generate Exercises", type="primary"):
        if not selected_docs:
            st.error("Please select at least one document.")
        else:
            initialize_sc_state()
            st.session_state.selected_docs_for_sc = selected_docs
            with st.spinner("Generating Exercises..."):
                vector_store = load_and_merge_vector_stores(user_data_path, selected_docs)
                if vector_store:
                    exercises = generate_fill_in_the_blank_exercises(vector_store, llm, num_exercises_to_gen)
                    if exercises:
                        st.session_state.sc_exercises = exercises
                        st.session_state.sc_started = True
                        st.rerun()
                    else:
                        st.error("Failed to generate exercises from the selected documents.")
                else:
                    st.error("Could not load the selected documents.")

else: # Gameplay is active
    st.markdown("---")
    st.info(f"Exercise based on: {', '.join(st.session_state.selected_docs_for_sc)}")
    current_index = st.session_state.sc_current_index
    ex_data = st.session_state.sc_exercises[current_index]

    st.subheader(f"Exercise {current_index + 1} of {len(st.session_state.sc_exercises)}")
    st.markdown(f" {ex_data['sentence_with_blank'].replace('[BLANK]', '______')}")

    st.text_input(
        "Your answer:",
        key=f"text_{current_index}",
        value=st.session_state.sc_user_answers.get(current_index, "")
    )
    # Update answer in session state
    st.session_state.sc_user_answers[current_index] = st.session_state[f"text_{current_index}"]

    st.markdown("---")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if current_index > 0:
            if st.button("⬅️ Previous"):
                st.session_state.sc_current_index -= 1
                st.rerun()
    with col2:
        if current_index < len(st.session_state.sc_exercises) - 1:
            if st.button("Next ➡️"):
                st.session_state.sc_current_index += 1
                st.rerun()
    with col3:
        if st.button("Submit All Answers", type="primary"):
            st.session_state.sc_submitted = True
            final_score = 0
            for i, ex in enumerate(st.session_state.sc_exercises):
                if st.session_state.sc_user_answers.get(i, "").strip().lower() == ex['correct_answer'].strip().lower():
                    final_score += 1
            save_user_score(user_data_path, final_score, len(st.session_state.sc_exercises))
            st.rerun()
    with col4:
        if st.button("Try Another Set", type="secondary"):
            initialize_sc_state()
            st.rerun()
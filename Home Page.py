import streamlit as st
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Scholaris - Your Learning Companion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MAIN PAGE UI ---
st.title("📚 Scholaris - Your Learning Companion")
st.sidebar.success("Select a learning mode from the pages listed here.")

st.markdown(
    """
    Welcome! This app transforms your documents into interactive learning exercises. 
    To get started, please log in or sign up below.
    """
)

def show_authenticated_content():
    st.session_state["user_name"] = st.user.get("nickname")
    st.session_state["user_email"] = st.user.get("email")
    st.session_state["user_id"] = st.user.get("sub")
    # --- USER INPUT & CACHING ---
    st.info("Your API key is used for this session only and is not stored.", icon="ℹ️")

    user_name = st.session_state["user_name"]

    # --- DIRECTORY SETUP ---
    if not os.path.exists('user_data'):
        os.makedirs('user_data')

    google_api_key = st.text_input(
        "Enter your Google API Key", 
        type="password", 
        key="api_key_main_input",
        value=st.session_state.get("google_api_key", "") # Persist value
    )

    if user_name and google_api_key:
        st.session_state.user_name = user_name
        st.session_state.google_api_key = google_api_key
        st.success(f"Welcome, **{user_name}**! Your details are set for this session. Please proceed to the other pages.", icon="✅")
    else:
        st.warning("Please enter your Google API Key to proceed.", icon="⚠️")


    st.markdown(
        """
        ---
        ### How It Works:

        1.  **Enter Details**: Provide your Google API key above.
        2.  **Upload & Process**: Navigate to the **Upload & Process** page to upload your documents.
        3.  **Learn & Practice**: Go to the **Multiple Choice Quiz**, **Sentence Completion** or **Question Answering** pages to start practicing.
        4.  **Track Your Progress**: Visit the **User Dashboard** to see your scores.

        ### Core Learning Techniques Used:

        * **Active Recall**: By asking you to retrieve information, the app forces your brain to work harder, strengthening memory pathways.
        * **Contextual Learning**: All questions are generated directly from your uploaded documents.
        * **Instant Feedback**: Receive immediate corrections and explanations.
        """
    )
    st.markdown("---")
    st.sidebar.button("Logout", on_click=st.logout)

if st.user.is_logged_in:
    show_authenticated_content()
else:
    st.button("Log In / Sign Up", on_click=st.login, args=["auth0"])
    st.stop()

    
# **📚 Scholaris \- Your Personal AI Learning Companion**

Scholaris is an agentic AI web application designed to transform passive reading into active, effective learning. It takes your personal documents, lecture notes, or academic papers and converts them into a suite of interactive exercises to help you master any subject.

## **✨ Core Features**

* **Secure User Management:** Full authentication system powered by **Auth0**, ensuring each user's data is private and secure.  
* **Dynamic Document Library:**  
  * **File Uploads:** Upload your own PDF, DOCX, or TXT files.  
  * **Arxiv Integration:** Search for and process the latest academic papers directly from Arxiv without leaving the app.  
  * **Library Management:** Easily view and remove processed documents from your personal library.  
* **Interactive Learning Modules:**  
  * **Multiple Choice Quizzes:** Generate custom quizzes based on one or more documents, with instant feedback and source references.  
  * **Sentence Completion:** Test your recall of key terms and phrases from your material.  
  * **AI-Graded Q\&A:** Answer open-ended questions generated from your documents and receive a semantic similarity score against an AI-generated ideal answer.  
  * **AI-Powered Summarization:** Create concise, markdown-formatted summaries of dense documents and save them back to your library to enrich the source material.  
* **Personalized Progress Tracking:**  
  * **User Dashboard:** View detailed metrics on your performance, including average scores and total exercises completed.  
  * **GitHub-Style Activity Calendar:** Visualize your daily study performance and track your learning streaks.

## **🚀 How It Works**

1. **Log In:** Create an account or log in securely using Auth0.  
2. **Provide API Key:** Enter your Google API Key for the session to enable the AI features (To ensure transparency, the API Key is retained only in the active user's session)
3. **Build Your Library:** Navigate to the **Library Management** page to upload your own files or search Arxiv for relevant papers.  
4. **Generate Exercises:** Go to any of the learning modules (e.g., Multiple Choice Quiz), select the document(s) you want to study, and generate a new exercise set.  
5. **Learn & Review:** Complete the exercises, get instant feedback and scores, and track your overall progress on the dashboard.

## **🛠️ Tech Stack**

* **Frontend:** [Streamlit](https://streamlit.io/)  
* **Backend / AI Orchestration:** [LangChain](https://www.langchain.com/)  
* **Generative AI:** [Google Gemini](https://ai.google.dev/)  
* **Authentication:** [Auth0](https://auth0.com/)  
* **Vector Embeddings:** [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
* **Vector Storage:** [FAISS](https://github.com/facebookresearch/faiss) (local storage)  
* **Academic Papers:** [Arxiv Python Library](https://github.com/lukasschwab/arxiv.py)

## **⚙️ Local Setup and Installation**

Follow these steps to get a local copy up and running.

### **Prerequisites**

* Python 3.9+  
* An [Auth0](https://auth0.com) account  
* A [Google API Key](https://ai.google.dev/gemini-api/docs/api-key)

### **Installation**

1. **Clone the repository:**  
   ```
   git clone https://github.com/your-username/scholaris.git  
   cd scholaris
   ```

2. **Create a virtual environment:**  
   ```
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`
   ```

3. **Install dependencies:**  
   ```
   pip install \-r requirements.txt
   ```

4. **Configure your secrets:**  
   * Create a file at .streamlit/secrets.toml.  
   * Add your credentials in the following format:  
     ```
     \[auth\]  
     redirect\_uri \= "http://localhost:8501"  
     logout\_uri \= "http://localhost:8501" \# Or your Auth0 logout URL

     \[auth.auth0\]  
     domain \= "YOUR\_AUTH0\_DOMAIN"  
     client\_id \= "YOUR\_AUTH0\_CLIENT\_ID"  
     client\_secret \= "YOUR\_AUTH0\_CLIENT\_SECRET"
     ```

5. **Configure your Auth0 Application:**  
   * In your Auth0 dashboard, ensure that http://localhost:8501 is added to both **Allowed Callback URLs** and **Allowed Logout URLs**.  
6. **Run the application:**  
   ```
   streamlit run Home\ Page.py
   ```

## **🗺️ Future Roadmap**

* **Cloud Backend Integration:** Migrate from local file storage to a robust cloud backend like **Supabase** or a combination of a vector database like **Qdrant** and a SQL database.  
* **Advanced Learning Modules:**  
  * Implement an automated **Flashcard Deck** with a Spaced Repetition System (SRS).  
  * Add **Concept Mapping** to visually connect key ideas within documents.
* **Fine-tuned Models:** Better models used on specific exercises  
* **Agent with Tools-calling:** Implement tools-calling to better handle documents processing, information searching

## **🤝 Contributing**

Contributions, issues, and feature requests are welcome\! Feel free to check the issues page.

## **📜 License**

Distributed under the MIT License. See LICENSE.txt for more information.
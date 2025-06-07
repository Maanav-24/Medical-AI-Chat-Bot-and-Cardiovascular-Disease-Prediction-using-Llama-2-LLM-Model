## Medical AI Chatbot & Cardiovascular Disease Prediction

# 🚀 Project Overview

An intelligent, local-first healthcare assistant that combines natural language understanding with medical document analysis and disease risk prediction.

Built using LLaMA, LangChain, FAISS, and Chainlit, this system enables:

💬 Conversational Q&A — Chat with an AI-powered medical assistant trained on PDFs and structured health data.

📁 Smart Document Retrieval — Extracts and understands information from PDFs and CSVs for context-rich answers.

🧠 Local LLM Intelligence — Uses LLaMA with HuggingFace embeddings and sentence transformers to generate accurate, context-aware responses.

🩺 Health Risk Prediction — Capable of integrating models to assess cardiovascular risk from clinical input.

⚡ Real-Time Interaction — Engage with the chatbot through a sleek Chainlit interface.

All components are designed to work locally, ensuring privacy and full control over the pipeline.


## 🧰 Tech Stack

- **🦙 LLaMA (via CTransformers)** – Lightweight local LLM for generating responses.
- **🧠 LangChain** – Framework to build retrieval-augmented generation pipelines.
- **📚 FAISS** – Vector store for efficient semantic search on embedded documents.
- **🔤 HuggingFace Embeddings** – Converts text into dense vectors for similarity comparison.
- **💬 Sentence Transformers** – High-quality sentence-level embeddings (`all-MiniLM-L6-v2`).
- **📄 PyPDFLoader & DirectoryLoader** – For loading and processing PDF and text documents.
- **📊 XGBoost** – Gradient boosting classifier for cardiovascular disease risk prediction.
- **🧮 Pandas** – Used for reading and preprocessing structured health data (CSV).
- **🌐 Chainlit** – Real-time web interface for conversational chatbot interactions.
- **🐍 Python** – Core language used for backend logic and model orchestration.


## ⚙️ How it works:

1. Preprocessing: CSV files are read and converted into plain text format. PDFs are also loaded.

2. Embedding Generation: Text chunks are created using a recursive splitter and embedded using Sentence Transformers.

3. Vector Store Creation: FAISS stores these embeddings for efficient retrieval during Q&A.

4. Chat Pipeline:

    a. On query, the system retrieves the most relevant chunks from the vector store.
    b. These chunks are passed to the LLaMA model with a custom prompt.
    c. A final answer is generated and returned through Chainlit's interface.

5. Live Interaction: Chainlit provides real-time communication between the user and the AI system.


## 🚀 Getting Started:

1. Place all relevant PDFs and CSVs inside your dataset directory:
C:/MEDIBOT/dataset/

2. Run the script to generate the FAISS vector store.

3. Start the chatbot and interact through the Chainlit UI.


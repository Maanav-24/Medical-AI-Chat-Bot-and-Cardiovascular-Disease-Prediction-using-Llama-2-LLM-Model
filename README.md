                               Medical AI Chatbot & Cardiovascular Disease Prediction

# OVERVIEW:

An intelligent, local-first healthcare assistant that combines natural language understanding with medical document analysis and disease risk prediction.

Built using LLaMA, LangChain, FAISS, and Chainlit, this system enables:

💬 Conversational Q&A — Chat with an AI-powered medical assistant trained on PDFs and structured health data.

📁 Smart Document Retrieval — Extracts and understands information from PDFs and CSVs for context-rich answers.

🧠 Local LLM Intelligence — Uses LLaMA with HuggingFace embeddings and sentence transformers to generate accurate, context-aware responses.

🩺 Health Risk Prediction — Capable of integrating models to assess cardiovascular risk from clinical input.

⚡ Real-Time Interaction — Engage with the chatbot through a sleek Chainlit interface.

All components are designed to work locally, ensuring privacy and full control over the pipeline.


## 🧰 TECH STACK:

- **🦙 LLaMA (via CTransformers)** – Lightweight local LLM for generating answers.
- **🧠 LangChain** – Framework for building retrieval-based QA pipelines.
- **📚 FAISS** – Vector store for efficient similarity search on embedded text chunks.
- **🔤 HuggingFace Embeddings** – Transforms documents into semantic vectors.
- **💬 Sentence Transformers** – Used for high-quality sentence-level embeddings.
- **📄 PyPDFLoader & DirectoryLoader** – For loading PDF and CSV (as text) documents.
- **🌐 Chainlit** – Lightweight UI framework for real-time chatbot interaction.
- **🐼 Pandas** – For reading and processing structured CSV data.
- **📦 Python** – Core language for backend logic and integration.


## ⚙️ HOW IT WORKS:

1. Preprocessing: CSV files are read and converted into plain text format. PDFs are also loaded.

2. Embedding Generation: Text chunks are created using a recursive splitter and embedded using Sentence Transformers.

3. Vector Store Creation: FAISS stores these embeddings for efficient retrieval during Q&A.

4. Chat Pipeline:

    a. On query, the system retrieves the most relevant chunks from the vector store.
    b. These chunks are passed to the LLaMA model with a custom prompt.
    c. A final answer is generated and returned through Chainlit's interface.

5. Live Interaction: Chainlit provides real-time communication between the user and the AI system.


## 🚀 GETTING STARTED:

1. Place all relevant PDFs and CSVs inside your dataset directory:
C:/MEDIBOT/dataset/

2. Run the script to generate the FAISS vector store.

3. Start the chatbot and interact through the Chainlit UI.


import os
import logging
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_PATH2 = 'C:/MEDIBOT/dataset'  
DB_FAISS_PATH2 = 'C:/MEDIBOT/vectorstore/db_faiss'

def preprocess_csv(csv_path):
    """Preprocesses CSV data and converts it to text."""
    try:
        df = pd.read_csv(csv_path)
        text_data = df.to_string()  
        text_file_path = csv_path.replace('.csv', '.txt')
        with open(text_file_path, 'w') as file:
            file.write(text_data)
        return text_file_path
    except Exception as e:
        logger.error(f"Error preprocessing CSV file {csv_path}: {e}")
        return None

# Create vector database
def create_vector_db():
    try:
        # Handle PDF files
        pdf_loader = DirectoryLoader(DATA_PATH2, glob='*.pdf', loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        
        # Handle CSV files
        csv_files = [f for f in os.listdir(DATA_PATH2) if f.endswith('.csv')]
        csv_text_files = []
        for csv_file in csv_files:
            csv_path = os.path.join(DATA_PATH2, csv_file)
            text_file_path = preprocess_csv(csv_path)
            if text_file_path:
                csv_text_files.append(text_file_path)
        
        # Load text files
        text_files_loader = DirectoryLoader(DATA_PATH2, glob='*.txt', loader_cls=PyPDFLoader)
        text_documents = text_files_loader.load()
        
        # Combine PDF and CSV text documents
        documents = pdf_documents + text_documents
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        
        # Create and save FAISS database
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH2)
        logger.info("Vector database created and saved successfully.")
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")

def set_custom_prompt():
    """Returns a custom prompt template for QA retrieval."""
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that I don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def retrieval_qa_chain(llm, prompt, db):
    """Returns a RetrievalQA chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    """Loads the local LLM model."""
    return CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

def qa_bot():
    """Sets up and returns the QA bot."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH2, embeddings, allow_dangerous_deserialization=True)
        llm = load_llm()
        qa_prompt = set_custom_prompt()
        return retrieval_qa_chain(llm, qa_prompt, db)
    except Exception as e:
        logger.error(f"Error setting up QA bot: {e}")

def final_result(query):
    """Gets the final result for a query."""
    try:
        qa_result = qa_bot()
        response = qa_result({'query': query})
        return response
    except Exception as e:
        logger.error(f"Error getting final result: {e}")
        return {"result": "An error occurred while processing your query."}

@cl.on_chat_start
async def start():
    """Handles the start of the chat."""
    try:
        chain = qa_bot()
        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, Welcome to Medical Chat Bot. What is your query?"
        await msg.update()
        cl.user_session.set("chain", chain)
    except Exception as e:
        logger.error(f"Error starting chat: {e}")

@cl.on_message
async def main(message: cl.Message):
    """Handles incoming messages and provides responses."""
    try:
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res.get("source_documents", [])
        await cl.Message(content=answer).send()
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await cl.Message(content="An error occurred while processing your message.").send()

# Test the chatbot
def test_chatbot(query):
    try:
        response = final_result(query)
        print(f"Query: {query}")
        print(f"Response: {response['result']}")
        print(f"Sources: {response.get('source_documents', 'No sources found')}")
    except Exception as e:
        logger.error(f"Error during testing: {e}")

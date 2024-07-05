import os

# Langchain libraries for building the retrieval-based QA pipeline
from langchain import HuggingFaceHub  # Access pre-trained models from Hugging Face Hub
from langchain.chains import RetrievalQA  # Build a retrieval-based question answering pipeline
from langchain.document_loaders import TextLoader  # Load documents from text files
from langchain.embeddings import HuggingFaceEmbeddings  # Generate embeddings for documents using Hugging Face models
from langchain.text_splitter import CharacterTextSplitter  # Split documents into smaller chunks for processing
from langchain.vectorstores import FAISS  # Use FAISS for efficient retrieval of similar documents

from transformers import pipeline  # Load pre-trained models for text generation from Transformers library

class CustomerAssistanceAgent():
    # Class implements Customer Assistance Agent
    def __init__(self):
        # Define Parameters
        self.repo_id = "google/flan-t5-large"
        self.model_kwargs = {"temperature":0, "max_length":64}
        self.data_path = './data/demo.txt'
        self.chunk_size = 1000
        self.chunk_overlap = 0
        self.db_path = "faiss_index"

        # Loads entire pipline
        self.load_pipline()
    
    def load_pipline(self):
        # Load the Large Language Model (LLM)
        llm = HuggingFaceHub(repo_id=self.repo_id, model_kwargs=self.model_kwargs)

        # Load the documents
        loader = TextLoader(self.data_path)
        documents = loader.load()

        # Split the documents into smaller chunks (optional)
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)

        # Generate document embeddings
        embeddings = HuggingFaceEmbeddings()

        # Create a FAISS index for efficient retrieval
        db = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index (optional)
        db.save_local(self.db_path)

        # Load the FAISS index (if saved previously)
        new_db = FAISS.load_local(self.db_path, embeddings)

        # Create a retriever object
        retriever = new_db.as_retriever()

        # Build the RetrievalQA pipeline
        qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)
        return qa_stuff

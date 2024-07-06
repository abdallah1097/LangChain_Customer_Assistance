import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

import logging
logging.captureWarnings(True) # Ignore all warnings

# Langchain libraries for building the retrieval-based QA pipeline
from langchain_community.llms import HuggingFaceHub # Access pre-trained models from Hugging Face Hub
from langchain.chains import RetrievalQA  # Build a retrieval-based question answering pipeline
from langchain_community.document_loaders import TextLoader  # Load documents from text files
from langchain_community.embeddings import HuggingFaceEmbeddings  # Generate embeddings for documents using Hugging Face models
from langchain.text_splitter import CharacterTextSplitter  # Split documents into smaller chunks for processing
from langchain_community.vectorstores import FAISS  # Use FAISS for efficient retrieval of similar documents


class CustomerAssistanceAgent():
    # Class implements Customer Assistance Agent
    def __init__(self):
        # Define Parameters
        self.repo_id = "google/flan-t5-large"
        self.model_kwargs = {"temperature": 0.7, "max_new_tokens": 250, "top_k":50, "repetition_penalty":1.03}
        self.data_path = './data/menu.txt'
        self.chunk_size = 1000
        self.chunk_overlap = 0
        self.db_path = "faiss_index"

        # Loads entire pipline
        self.pipeline = self.load_pipline()

        # Get answer format
        self.prompt = self.get_answer_format()

    def load_pipline(self):
        """
        Loads and builds the retrieval-based question answering pipeline.

        Returns:
            RetrievalQA: The constructed retrieval-based question answering pipeline.
        """
        # Load the Large Language Model (LLM)
        llm = HuggingFaceHub(repo_id=self.repo_id, model_kwargs=self.model_kwargs)

        # Load the documents
        loader = TextLoader(self.data_path)
        documents = loader.load()

        # Split the documents into smaller chunks (optional)
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)

        # See the splitted text document
        print(f"Data Document Splitted into: {len(docs)} Chuncks {len(docs[0].page_content)} Char Each!")
        for i, doc in enumerate(docs, 0):
            print(f"    Chunck [{i}]: {len(doc.page_content)} Char | Starts with:\n        {doc.page_content[:50]}")

        # Generate document embeddings
        embeddings = HuggingFaceEmbeddings()

        # Create a FAISS index for efficient retrieval
        db = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index (optional)
        db.save_local(self.db_path)

        # Load the FAISS index (if saved previously)
        new_db = FAISS.load_local(self.db_path, embeddings, allow_dangerous_deserialization=True)

        # Create a retriever object
        retriever = new_db.as_retriever()

        # Build the RetrievalQA pipeline
        qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=False)
        return qa_stuff

    def get_answer_format(self):
        """
        Generates a formatted string template for presenting the user's question and the answer.

        This function constructs a template string with placeholders for the question (`{question}`)
        and the answer (`{answer}`). This template is then used to format the response returned
        by the retrieval-based QA pipeline.

        Returns:
            str: The formatted template string.
        """
        template = """
        Question: {question}
        Answer for the given documents:
            {answer}"""

        return template.format(question="{question}", answer="{answer}")

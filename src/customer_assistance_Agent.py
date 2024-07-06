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
from sentence_transformers import SentenceTransformer

class CustomerAssistanceAgent():
    # Class implements Customer Assistance Agent
    def __init__(self):
        # Define Parameters
        self.repo_id = "google/flan-t5-large"
        self.model_kwargs = {"temperature": 0.7, "max_new_tokens": 250, "top_k":50, "repetition_penalty":1.03}
        self.embedding_model_kargs = {}
        self.data_path = './data/menu.txt'
        self.chunk_size = 500
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
        llm = self.load_llm()

        # Load the documents
        docs = self.load_split_text()

        # Generate document embeddings
        sentence_embeddings = self.generate_embeddings(docs)

        # Create a FAISS index for efficient retrieval
        db = FAISS.from_documents(docs, sentence_embeddings)

        # Save the FAISS index (optional)
        db.save_local(self.db_path)

        # Load the FAISS index (if saved previously)
        new_db = FAISS.load_local(self.db_path, embeddings, allow_dangerous_deserialization=True)

        # Create a retriever object
        retriever = new_db.as_retriever()

        # Build the RetrievalQA pipeline
        qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=False)
        return qa_stuff


    def load_llm(self):
        """
        Loads the Large Language Model (LLM)
        """
        return HuggingFaceHub(repo_id=self.repo_id, model_kwargs=self.model_kwargs)

    def load_split_text(self):
        """
        Loads text dataset from attribute self.data_path and return splitted docs
        """
        loader = TextLoader(self.data_path)
        documents = loader.load()

        # Split the documents into smaller chunks (optional)
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splitted_documents = text_splitter.split_documents(documents)

        docs = [i.page_content for i in splitted_documents]

        # See the splitted text document
        print(f"Data Document Splitted into: {len(docs)} Chuncks {len(docs[0])} Char Each!")
        for i, doc in enumerate(docs, 0):
            print(f"    Chunck [{i}]: {len(doc)} Char | Starts with:\n        {doc[:50]}")
        return docs

    def generate_embeddings(self, docs):
        """
        Generates model embeddings
        """
        # embeddings = HuggingFaceEmbeddings()
        embeddings_model = SentenceTransformer('bert-base-nli-mean-tokens')
        # create sentence embeddings
        sentence_embeddings = embeddings_model.encode(docs)
        print(f"Sentence Embeddings Shape: {sentence_embeddings.shape}")
        return sentence_embeddings

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

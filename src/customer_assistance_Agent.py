import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# import logging
# logging.captureWarnings(True) # Ignore all warnings

# Langchain libraries for building the retrieval-based QA pipeline
from langchain_community.document_loaders.generic import GenericLoader  # Load documents from text files
from langchain_community.embeddings import HuggingFaceEmbeddings  # Generate embeddings for documents using Hugging Face models
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub # Access pre-trained models from Hugging Face Hub
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
        self.repo_id = "openai-community/gpt2" # "distilbert/distilgpt2" # "facebook/opt-125m" # "openai-community/gpt2" # "google/flan-t5-large"
        self.model_kwargs = {"temperature": 0.1, "max_new_tokens": 250, "top_k": 1, "repetition_penalty":1.03}
        self.embedding_model_kargs = {}
        self.data_path = './data'
        self.db_path = "faiss_index"

        # Initialize Models
        self.embeddings_model = HuggingFaceEmbeddings()
        # SentenceTransformer(model_name_or_path='all-MiniLM-L12-v2',
                                                    # similarity_fn_name='cosine',
                                                    # )
        self.llm_model = HuggingFaceHub(repo_id="openai-community/gpt2", model_kwargs=self.model_kwargs)

        # Load the documents
        docs = self.load_split_text()

        # Loads entire pipline
        self.pipeline = self.load_pipline(docs)

        # Get answer format
        self.prompt = self.get_answer_format()

    def load_pipline(self, docs):
        """
        Loads and builds the retrieval-based question answering pipeline.

        Returns:
            RetrievalQA: The constructed retrieval-based question answering pipeline.
        """


        # Create a FAISS index for efficient retrieval
        db, retriever = self.create_faiss_db(docs)

        # Build the RetrievalQA pipeline
        qa_stuff = RetrievalQA.from_chain_type(llm=self.llm_model, chain_type="stuff", retriever=retriever, verbose=True)
        return qa_stuff

    def load_split_text(self):
        """
        Loads text dataset from attribute self.data_path and return splitted docs
        """
        # Create a GenericLoader instance specifying the directory and file extension
        loader = GenericLoader.from_filesystem(path=self.data_path, suffixes=[".txt"])

        # This loads all the .txt files in a lazy manner (loads them on demand)
        documents = loader.lazy_load()

        # Split the documents into smaller chunks (optional)
        text_splitter = CharacterTextSplitter()
        splitted_document = text_splitter.split_documents(documents)

        # See the splitted text document
        print(f"\n[INFO] Data Document Splitted into: {len(splitted_document)} Chuncks {len(splitted_document[0].page_content)} Char Each!")
        for i, doc in enumerate(splitted_document, 0):
            print(f"    Chunck [{i}]: {len(doc.page_content)} Char | Starts with:\n        {doc.page_content}")
        return splitted_document

    def create_faiss_db(self, docs):
        db = FAISS.from_documents(docs, self.embeddings_model)

        # Create a retriever object
        retriever = db.as_retriever(
                # search_type="similarity_score_threshold",
                # search_kwargs={'score_threshold': 0.5}
                search_type="similarity",
                search_kwargs={'k': 5}
                # search_type="mmr",
                # search_kwargs={'k': 5, 'fetch_k': 71}
            )
        print(f"\n[INFO] Created Index with: {db.index.ntotal}. Trying query database:")

        query = "What rice you have?"
        responses = retriever.invoke(query)
        for response in responses:
            print(f"\n\n    query: {query} response: {response.page_content}")

        return db, retriever

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
        This is what we got in our meneu:
            {answer}"""

        return template.format(question="{question}", answer="{answer}")

    def query_with_prefix(self, question):
        """
        Adds a prefix to the LLM prompt after retrieving relevant documents based on the original query.

        Args:
            question (str): The original query.

        Returns:
            str: The response from the LLM model with the prefixed query.
        """
        # Retrieve relevant documents based on the original query
        retriever = self.pipeline.retriever
        retrieved_docs = retriever.invoke(question)

        # Combine the retrieved documents into a single context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prefix_1 = "Agent requested this query: "
        prefix_2 = "This is our menue:"
        prefix_3 = "reply to user in a fancy human-like way"

        # Add the prefix to the LLM prompt
        llm_prompt = f"{prefix_1}{question}\n\n{prefix_2}\n{context}\n\n{prefix_3}"
        print(f"\n\nllm_prompt: {llm_prompt}")

        # Get the response from the LLM model
        response = self.llm_model(llm_prompt)
        return response
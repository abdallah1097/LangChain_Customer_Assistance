import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
logging.captureWarnings(True) # Ignore all warnings

# Langchain libraries for building the retrieval-based QA pipeline
from langchain_community.document_loaders.generic import GenericLoader  # Load documents from text files
from langchain_community.llms import HuggingFaceHub # Access pre-trained models from Hugging Face Hub
from langchain.chains import RetrievalQA  # Build a retrieval-based question answering pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings  # Generate embeddings for documents using Hugging Face models
from langchain.text_splitter import CharacterTextSplitter  # Split documents into smaller chunks for processing
from langchain_community.vectorstores import FAISS  # Use FAISS for efficient retrieval of similar documents
from g4f.client import Client


class CustomerAssistanceAgent():
    # Class implements Customer Assistance Agent
    def __init__(self):
        # Define Parameters
        self.llm_model_name = "gpt-3.5-turbo"
        self.model_kwargs = {"temperature": 0.5, "max_new_tokens": 250, "top_k": 1}
        self.embedding_model_kargs = {}
        self.data_path = './data'
        self.DEBUG = True

        # Initialize Models
        self.embeddings_model = HuggingFaceEmbeddings()

        # Initialize the GPT client with the desired provider
        self.gpt_client = Client()

        # Create a memory of appended chat history
        self.chat_history = []

        # Load the documents
        docs = self.load_split_text()

        # Create a FAISS index for efficient retrieval
        self.db, self.retriever = self.create_faiss_db(docs)

        # Get answer format
        self.prompt = self.get_answer_format()

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

        if self.DEBUG:
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
        if self.DEBUG:
            print(f"\n[INFO] Created Index with: {db.index.ntotal}. Trying query database:")

        query = "What rice you have?"
        responses = retriever.invoke(query)
        if self.DEBUG:
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
        # Update the conversation history with GPT's response
        self.chat_history.append({"role": "user", "content": question})

        question_llm_prompt = f"Extract important keywords, ignore all questions words/ irrelevent content and return ONLY KEYWORDS: {question}"
        gpt_question_response = "sorry" # To keep retrying from agent

        while "sorry" in gpt_question_response:
            # Get GPT's response
            response = self.gpt_client.chat.completions.create(
                messages=[{"role": "user", "content": question_llm_prompt}],
                model=self.llm_model_name,
            )

            # Extract the GPT response and print it
            gpt_question_response = response.choices[0].message.content
        if self.DEBUG:
            print(f"    [INFO] Summarized Response: {gpt_question_response}")

        # Retrieve relevant documents based on the original query
        retriever = self.retriever
        retrieved_docs = retriever.invoke(gpt_question_response)

        # Combine the retrieved documents into a single context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        if self.DEBUG:
            print(f"    [INFO] Database hit return: {context}")

        # Add the prefix to the LLM prompt
        llm_prompt = f"Agent requested this query: {question}\n\nThis is our menue: \n{context}\n\nSummarize only {gpt_question_response} dishes found in menue and ignore any other meals doesn't have {gpt_question_response} in a fancy human-like way"
        gpt_final_response = "sorry"
        if self.DEBUG:
            print(f"    [INFO] Final LLM Prompt: {llm_prompt}")
        while "sorry" in gpt_final_response:
            # Get GPT's response
            response = self.gpt_client.chat.completions.create(
                messages=[{"role": "user", "content": llm_prompt}],
                model=self.llm_model_name,
            )

            # Extract the GPT response and print it
            gpt_final_response = response.choices[0].message.content

        print(f"Bot: {gpt_final_response}")

        # Update the conversation history with GPT's response
        self.chat_history.append({"role": "assistant", "content": gpt_final_response})

        return gpt_final_response
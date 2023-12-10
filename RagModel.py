from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
api_key = os.environ["OPENAI_API_KEY"]

# Create OpenAI language model
llm = OpenAI(openai_api_key=api_key, temperature=0.9, max_tokens=500)

# Initialize embeddings using the Hugging Face model
embeddings = OpenAIEmbeddings()

# Define the file path for the FAISS vector database
vectordb_file_path = "./data/faiss_index"

def create_vector_db():
    # Create the directory if it doesn't exist
    os.makedirs("data/faiss_index", exist_ok=True)

    # Load data from the PDF document
    loader = PyPDFLoader("./data/products.pdf")
    data = loader.load()

    # Create a FAISS instance for the vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)

    # Save the vector database locally
    vectordb.save_local(vectordb_file_path)

def getChain():
    try:
        # Load the vector database from the local folder
        vectordb = FAISS.load_local(vectordb_file_path, embeddings)

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.7)

        # Define the prompt template
        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        # Create a prompt template instance
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create a RetrievalQA chain
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            input_key="query",
                                            return_source_documents=False,
                                            chain_type_kwargs={"prompt": PROMPT})

        return chain

    except Exception as e:
        # Handle exceptions gracefully and print an error message
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    try:
        # Create the vector database
        create_vector_db()

        # Get the RetrievalQA chain
        chain = getChain()

        # Check if chain is not None before using it
        if chain:
            # Perform a sample query
            result = chain("Hi there! I'm looking for a new pair of running shoes. Can you help me find some options?")
            print(result)

    except Exception as e:
        # Handle exceptions gracefully and print an error message
        print(f"An error occurred: {e}")

import logging
import time

import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Ensure your API key is securely managed
OPENAI_API_KEY = ""  # Use environment variables in production

# Setup logging for detailed error information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit caching for FAISS index
@st.cache_resource(show_spinner=False)
def create_faiss_index(text_chunks, api_key, retries=10, backoff_factor=2):
    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt + 1} of {retries} to generate embeddings.")
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            # Create FAISS index from text chunks and embeddings
            return FAISS.from_texts(text_chunks, embeddings)
        except openai.RateLimitError as e:
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                st.warning(f"Rate limit exceeded: {e}. Retrying in {sleep_time} seconds...")
                logger.warning(f"Rate limit exceeded: {e}. Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
            else:
                st.error(f"Rate limit exceeded after {retries} retries: {e}")
                logger.error(f"Rate limit exceeded after {retries} retries: {e}")
                raise
        except openai.InvalidRequestError as e:
            st.error(f"Invalid request: {e}")
            logger.error(f"Invalid request: {e}")
            raise
        except openai.AuthenticationError as e:
            st.error(f"Authentication error: {e}")
            logger.error(f"Authentication error: {e}")
            raise
        except openai.OpenAIError as e:
            st.error(f"An OpenAI error occurred: {e}")
            logger.error(f"An OpenAI error occurred: {e}")
            raise
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"An unexpected error occurred: {e}")
            raise

# Streamlit interface
st.header("My First Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract and process the text from the PDF
if file is not None:
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if text.strip():
            # Break text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n"],  # List of strings for separators
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            try:
                # Create FAISS index with retry mechanism
                vector_store = create_faiss_index(chunks, OPENAI_API_KEY)

                st.success("Vector store created successfully.")
                user_question = st.text_input("Type Your question here")

                # do similarity search
                if user_question:
                    match = vector_store.similarity_search(user_question)
                    # st.write(match)

                    # define the LLM
                    llm = ChatOpenAI(
                        openai_api_key=OPENAI_API_KEY,
                        temperature=0,
                        max_tokens=1000,
                        model_name="gpt-3.5-turbo"
                    )

                    # output results
                    # chain -> take the question, get relevant document, pass it to the LLM, generate the output
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=match, question=user_question)
                    st.write(response)

                # Additional code to use the vector store, like querying it
                # e.g., perform searches using vector_store.search(query_embedding)
            except Exception as e:
                st.error(f"Failed to generate embeddings after retries: {e}")
                logger.error(f"Failed to generate embeddings after retries: {e}")
        else:
            st.warning("The uploaded PDF does not contain extractable text.")
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        logger.error(f"An error occurred while processing the PDF: {e}")

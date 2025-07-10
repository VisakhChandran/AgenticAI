import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
# Set your Google API Key as a Streamlit secret or environment variable
# Go to Streamlit -> Settings -> Secrets and add GOOGLE_API_KEY = "your_api_key_here"
# Or set it as an environment variable before running: export GOOGLE_API_KEY="your_api_key_here"

# Ensure the Google API Key is set
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it as a Streamlit secret or environment variable.")
    st.stop() # Stop execution if API key is missing

# --- Helper Functions ---

@st.cache_resource
def get_embeddings_model():
    """Caches and returns the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm_model():
    """Caches and returns the Google Generative AI model."""
    return ChatGoogleGenerativeAI(model='gemini-1.5-flash')

def format_docs(docs):
    """Formats a list of documents into a single string."""
    return "\n \n".join(doc.page_content for doc in docs)

def process_pdf_and_create_retriever(file_path):
    """
    Loads a PDF, chunks it, creates embeddings, and builds a FAISS vector store.
    Returns a retriever object.
    """
    st.info(f"Processing PDF: {os.path.basename(file_path)}...")

    # 1. Load the document
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Perform Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Hyper Parameter
        chunk_overlap=50  # Hyper Parameter
    )
    split_docs = splitter.split_documents(pages)
    st.success(f"Document split into {len(split_docs)} chunks.")

    # 3. Create Embeddings and Vector Store
    embeddings = get_embeddings_model()
    
    # Initialize FAISS with a dummy index for initial creation
    # The actual index will be built when documents are added
    # The dimension 384 comes from the 'all-MiniLM-L6-v2' model output dimension
    index = faiss.IndexFlatIP(384) 
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # Add documents to the vector store
    vector_store.add_documents(documents=split_docs)
    st.success("Documents embedded and stored in vector database.")

    # 4. Create retriever object
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}  # HyperParameter, specify the number of records
    )
    st.success("Retriever created successfully!")
    return retriever

# Function to reset the application state
def reset_application():
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.session_state.chat_history = []
    # Removed st.rerun() from here, as modifying session state should trigger a rerun naturally.

# --- Streamlit UI ---

st.set_page_config(page_title="Your PDF Chatbot", layout="centered")

st.title("📄 Your PDF Chatbot")
st.markdown("""
Upload a PDF document, and then ask questions about its content.
The application will use a Retrieval-Augmented Generation (RAG) model
to provide answers based on the document.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# Initialize session state for retriever, rag_chain, and chat history
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    st.write(f"Uploaded: {uploaded_file.name}")

    # Process PDF and create retriever only if not already done
    if st.session_state.retriever is None:
        with st.spinner("Processing document... This may take a moment."):
            try:
                st.session_state.retriever = process_pdf_and_create_retriever(temp_file_path)
                
                # Initialize RAG chain here, after retriever is ready
                prompt = hub.pull("rlm/rag-prompt")
                model = get_llm_model()
                st.session_state.rag_chain = (
                    {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                )
                st.success("Document processed and RAG system ready for questions!")
                st.session_state.chat_history = [] # Clear history for new document
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.session_state.retriever = None # Reset retriever on error
                st.session_state.rag_chain = None
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)
    else:
        st.success("Document already processed. You can ask questions!")

# Display chat history
st.markdown("---")
st.subheader("Chat History")
chat_container = st.container(height=300, border=True) # Fixed height container for chat history

with chat_container:
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history: # Display in chronological order
            with st.chat_message("user"):
                st.markdown(chat['question'])
            with st.chat_message("assistant"):
                st.markdown(chat['answer'])
    else:
        st.info("No questions asked yet. Start by typing in the chat box below.")

# Question Input using st.chat_input
if st.session_state.rag_chain:
    user_question = st.chat_input("Ask a question about the document:")

    if user_question:
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Add user question to history immediately
            st.session_state.chat_history.append({"question": user_question, "answer": "Thinking..."})
            # This st.rerun() is now essential to force the UI to update immediately
            # after the answer is ready, or after an error occurred.
            st.rerun() # Trigger a rerun to show "Thinking..." and then process

# This block will execute on the rerun triggered above
if st.session_state.rag_chain and st.session_state.chat_history and st.session_state.chat_history[-1]["answer"] == "Thinking...":
    last_question = st.session_state.chat_history[-1]["question"]
    try:
        # Generate answer (this now runs in the same script execution)
        with st.spinner("Generating answer..."): # Use spinner for visual feedback during processing
            answer = st.session_state.rag_chain.invoke(last_question)
        # Update the last entry with the actual answer
        st.session_state.chat_history[-1]["answer"] = answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        st.session_state.chat_history[-1]["answer"] = "Could not generate an answer due to an error."
    
    # This st.rerun() is now essential to force the UI to update immediately
    # after the answer is ready, or after an error occurred.
    st.rerun() # Trigger a rerun to show the final answer
else:
    # This ensures that if rag_chain is not ready or no question is pending,
    # the chat input is still displayed.
    pass


st.markdown("---")

# Reset button
st.button("Start New Chat / Reset", on_click=reset_application, help="Clear all data and start fresh with a new document or conversation.")
st.caption("To fully close the application, simply close this browser tab and stop the Streamlit process in your terminal.")

st.caption("Powered by LangChain, Streamlit, FAISS, HuggingFace Embeddings, and Google Gemini.")

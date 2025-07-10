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
# For local testing, you can uncomment the line below and replace with your key,
# but it's not recommended for deployment.
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# Ensure the Google API Key is set
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please set it as a Streamlit secret or environment variable.")
    st.stop()

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

# --- Streamlit UI ---

st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")

st.title("📄 RAG PDF Chatbot")
st.markdown("""
Upload a PDF document, and then ask questions about its content.
The application will use a Retrieval-Augmented Generation (RAG) model
to provide answers based on the document.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# Initialize session state for retriever and chat history
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    st.write(f"Uploaded: {uploaded_file.name}")

    if st.session_state.retriever is None:
        with st.spinner("Processing document... This may take a moment."):
            try:
                st.session_state.retriever = process_pdf_and_create_retriever(temp_file_path)
                st.success("Document processed and ready for questions!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.session_state.retriever = None # Reset retriever on error
            finally:
                # Clean up the temporary file
                os.remove(temp_file_path)
    else:
        st.success("Document already processed. You can ask questions!")

# Question Input
if st.session_state.retriever:
    question = st.text_input("Ask a question about the document:", key="question_input")

    if question:
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    # Building the RAG Prompt and Chain (re-evaluate each time for fresh prompt pull)
                    prompt = hub.pull("rlm/rag-prompt")
                    model = get_llm_model()

                    rag_chain = (
                        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | model
                        | StrOutputParser()
                    )

                    answer = rag_chain.invoke(question)
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                    st.success("Answer generated!")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    answer = "Could not generate an answer due to an error."
                
                # Display chat history
                for chat in reversed(st.session_state.chat_history):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
                    st.markdown("---")
else:
    st.info("Please upload a PDF document to get started.")

st.markdown("---")
st.caption("Powered by LangChain, Streamlit, FAISS, HuggingFace Embeddings, and Google Gemini.")


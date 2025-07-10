import streamlit as st
import os
import tempfile

# Importing libraries for RAG
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

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Application with PDF Upload", layout="centered")
st.title("📄 RAG Application with PDF Upload")
st.markdown("Upload a PDF and ask questions about its content!")

# --- Session State for RAG Components ---
# This helps persist components like vector_store across Streamlit reruns
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

# --- PDF Upload Section ---
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    st.info(f"Processing PDF: {uploaded_file.name}...")

    try:
        # --- RAG Pipeline Setup (from your code) ---
        # 1. Load Document
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # 2. Perform Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Hyper Parameter
            chunk_overlap=50  # Hyper Parameter
        )
        split_docs = splitter.split_documents(pages)

        # 3. Initialize Embeddings
        # Ensure you have 'sentence-transformers' installed: pip install sentence-transformers
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 4. Create FAISS Vector Store
        # FAISS index requires a dimension, 384 for all-MiniLM-L6-v2
        index = faiss.IndexFlatIP(384)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        # Add documents to Vector DB
        vector_store.add_documents(documents=split_docs)
        st.session_state.vector_store = vector_store

        # 5. Create Retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}  # HyperParameter, specify the number of records
        )
        st.session_state.retriever = retriever

        # 6. Initialize LLM
        # IMPORTANT: Replace 'YOUR_GOOGLE_API_KEY' with your actual Google API Key
        # You can set it as an environment variable or directly here.
        # Example: os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
        # For security, it's recommended to use Streamlit secrets or environment variables.
        # st.secrets["GOOGLE_API_KEY"] if using Streamlit secrets
        google_api_key = os.getenv("GOOGLE_API_KEY") # Or get from st.secrets
        if not google_api_key:
            st.warning("Google API Key not found. Please set the 'GOOGLE_API_KEY' environment variable or add it to Streamlit secrets.")
            st.stop() # Stop execution if API key is missing

        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=google_api_key)

        # 7. Build RAG Prompt and Chain
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n \n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        st.session_state.rag_chain = rag_chain

        st.success(f"PDF '{uploaded_file.name}' processed and RAG system ready!")

    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
    finally:
        # Clean up the temporary file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

# --- Question Input Section ---
if st.session_state.rag_chain:
    question = st.text_input("Ask a question about the document:", placeholder="e.g., What is a llama model?")

    if question:
        st.info(f"Asking: '{question}'...")
        try:
            with st.spinner("Generating answer..."):
                answer = st.session_state.rag_chain.invoke(question)
                st.subheader("Answer:")
                st.write(answer)
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")
else:
    st.info("Please upload a PDF document to get started.")


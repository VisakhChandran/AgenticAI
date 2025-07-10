import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# Load environment
# -----------------------------
load_dotenv(override=True)

# -----------------------------
# Configurations
# -----------------------------
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Setup Pinecone index
index_name = "agenticaipinecone"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Setup embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Setup LLM and prompt
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt = hub.pull("rlm/rag-prompt")

# -----------------------------
# Helper functions
# -----------------------------
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def load_to_vectordb(split_docs):
    vector_store.add_documents(documents=split_docs)

def format_docs(docs):
    return " \n\n".join(doc.page_content for doc in docs)

# Setup retriever and rag chain
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 RAG Document QA with Gemini + Pinecone")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded. Processing...")
    split_docs = process_pdf("uploaded.pdf")
    load_to_vectordb(split_docs)
    st.success(f"Document processed and loaded into vector DB ({len(split_docs)} chunks).")

question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(question)
    st.write("**Answer:**")
    st.write(answer)

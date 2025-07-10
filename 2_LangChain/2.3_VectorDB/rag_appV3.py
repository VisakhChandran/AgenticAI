import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import faiss

# Function to load and process the document
def load_and_process_document(FILE_PATH):
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index = faiss.IndexFlatIP(384)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=split_docs)
    return vector_store

# Function to create the RAG chain
def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain

# Streamlit UI
def main():
    st.title("RAG Application")
    st.write("Upload a PDF document and ask a question.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        FILE_PATH = "temp.pdf"
        with open(FILE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        vector_store = load_and_process_document(FILE_PATH)
        rag_chain = create_rag_chain(vector_store)

        question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if question:
                answer = rag_chain.invoke(question)
                st.write("Answer:")
                st.write(answer)
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()

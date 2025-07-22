import os
import tempfile
import operator
import streamlit as st
from typing import List, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Define the Pydantic class
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="Identified Topic from user input")
    Reasoning: str = Field(description="Rationale or Reason for selecting the topic from the user input")

# Attaching the Pydantic Class to Parser output
parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

# Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Function to format documents
def format_docs(docs):
    return "\n \n".join(doc.page_content for doc in docs)

# Function to load and process PDF files
def load_and_process_pdf(pdf_file):
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, pdf_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Load the PDF using the temporary file path
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Remove the temporary file and directory after loading
    os.remove(temp_file_path)
    os.rmdir(temp_dir)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    new_docs = text_splitter.split_documents(documents=docs)

    # Create a Chroma database and retriever
    db = Chroma.from_documents(new_docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    return retriever

# Define the functions for the nodes
def function_1(state: AgentState):
    print("------> Supervisor <-----------")
    question = state["messages"][-1]
    print("Question is : ", question)
    template = """
    Your task is to clarify if the given user query into the following categories : [USA, Not Related].
    Only respond with the Category Name and Nothing else.

    User question : {question}

    {format_instructions}
    """
    prompt = PromptTemplate(template=template, input_variables=["question"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | model | parser
    response = chain.invoke({"question": question})
    print("Parsed response from Supervisor : ", response)
    return {"messages": [response.Topic]}

def function_2(state: AgentState):
    print("============> RAG Node <======================")
    question = state["messages"][0]
    prompt = PromptTemplate(
        template="""
        You are an assistant for answering the given question and provide accurate answers.
        Use the following pieces retrieved from the context to answer the question.
        If you don't know the answer, then you just say that you don't know the answer.
        Use a maximum of three sentences to answer the question.

        \n
        Question : {question}

        \n
        Context : {context}

        \n
        Answer :
        """,
        input_variables=['context', 'question']
    )
    rag_chain = (
        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return {"messages": [result]}

def function_3(state: AgentState):
    print("===========> LLM Call <=====================")
    question = state["messages"][0]
    complete_query = "Answer the following question with your knowledge of the real world. The question is " + question
    response = model.invoke(complete_query)
    return {"messages": [response.content]}

# Define the router function
def router(state: AgentState):
    print("====================> Router <================")
    last_message = state["messages"][-1]
    print("Message received to Router ", last_message)
    if "usa" in last_message.lower():
        return "RAG Call"
    else:
        return "LLM Call"

# Create the workflow
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", function_1)
workflow.add_node("RAG", function_2)
workflow.add_node("LLM", function_3)
workflow.set_entry_point("Supervisor")
workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM"
    }
)
workflow.add_edge("RAG", END)
workflow.add_edge("LLM", END)
app = workflow.compile()

# Streamlit app
st.title("PDF Topic Identifier and Q&A")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# File uploader
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file is not None:
    st.session_state.retriever = load_and_process_pdf(pdf_file)
    st.success("PDF file loaded successfully!")

    # Identify the topic of the PDF
    topic_template = """
    Your task is to identify the main topic of the given PDF content.

    Content: {content}

    {format_instructions}
    """
    topic_prompt = PromptTemplate(template=topic_template, input_variables=["content"], partial_variables={"format_instructions": parser.get_format_instructions()})
    topic_chain = topic_prompt | model | parser

    # Save the uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, pdf_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Load the PDF content
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    content = "\n".join([doc.page_content for doc in docs])

    # Remove the temporary file and directory after loading
    os.remove(temp_file_path)
    os.rmdir(temp_dir)

    # Identify the topic
    topic_response = topic_chain.invoke({"content": content})
    st.write(f"Identified Topic: {topic_response.Topic}")
    st.write(f"Reasoning: {topic_response.Reasoning}")

    # User input for questions
    user_question = st.text_input("Ask a question about the PDF content:")

    if user_question:
        # Add user question to conversation history
        st.session_state.conversation_history.append(("User", user_question))

        # Get the answer
        result = app.invoke({"messages": [user_question]})

        # Add assistant response to conversation history
        st.session_state.conversation_history.append(("Assistant", result["messages"][-1]))

    # Display conversation history
    st.write("Conversation History:")
    for speaker, message in st.session_state.conversation_history:
        st.write(f"{speaker}: {message}")

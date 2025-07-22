import streamlit as st
import operator
from typing import List, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document class
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import torch # Import torch to check for CUDA availability

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Agentic RAG Chatbot", layout="centered")
st.title("Agentic RAG Chatbot with File Upload")
st.markdown("Upload a text file to provide context for RAG, then ask a question.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db" not in st.session_state:
    st.session_state.db = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "response_source" not in st.session_state:
    st.session_state.response_source = "N/A" # To track if response is from RAG or LLM

# --- Embedding Model and LLM Initialization ---
@st.cache_resource
def get_embeddings_model():
    """
    Caches the embedding model to avoid re-loading on every rerun.
    Explicitly sets the device to 'cpu' to prevent 'NotImplementedError'
    related to meta tensors on certain GPU setups or memory constraints.
    If you have a working CUDA setup and sufficient GPU memory, you can try
    device='cuda' or device='auto'.
    """
    # Determine the device to use. Prioritize CUDA if available, otherwise use CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.info(f"Loading embedding model on device: {device}")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en", model_kwargs={'device': device})

@st.cache_resource
def get_llm_model():
    """Caches the LLM model."""
    # The API key is automatically provided by the Canvas environment.
    # If running outside Canvas, you might need to set GOOGLE_API_KEY as an environment variable.
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

embeddings = get_embeddings_model()
model = get_llm_model()

# --- File Uploader for RAG Context ---
uploaded_file = st.file_uploader("Upload a text file for RAG context (.txt)", type=["txt"])

if uploaded_file is not None and st.session_state.db is None:
    # Read the content of the uploaded file
    file_content = uploaded_file.getvalue().decode("utf-8")
    st.info(f"Processing '{uploaded_file.name}' for RAG context...")

    # Create a Document object from the file content
    # The source metadata is important for tracking where the content came from
    docs = [Document(page_content=file_content, metadata={"source": uploaded_file.name})]

    # Splitting the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    new_docs = text_splitter.split_documents(documents=docs)

    # Creating DB Instance (Chroma)
    db = Chroma.from_documents(new_docs, embeddings)
    st.session_state.db = db

    # Creating a retriever object
    retriever = db.as_retriever(search_kwargs={"k": 3})
    st.session_state.retriever = retriever
    st.success(f"'{uploaded_file.name}' loaded and ready for RAG!")
elif uploaded_file is None and st.session_state.db is not None:
    st.info("Using previously uploaded file for RAG context.")
elif uploaded_file is not None and st.session_state.db is not None and uploaded_file.name != st.session_state.db.get().get('metadatas', [{}])[0].get('source'):
    # If a new file is uploaded, re-process
    st.session_state.db = None # Clear existing DB to force re-processing
    st.session_state.retriever = None
    st.experimental_rerun() # Rerun to process the new file


# --- LangChain/LangGraph Components ---

# Function to collect the text from the document retrieved from retriever
def format_docs(docs):
    return "\n \n".join(doc.page_content for doc in docs)

# Defining the Pydantic class
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="Identified Topic from user input")
    Reasoning: str = Field(description="Rationale or Reason for selecting the topic from the user input")

# Attaching the Pydantic Class to Parser output
parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

# Defining Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    response_source: str # Added to track the source of the response

# Defining the function (Supervisor)
def function_1(state: AgentState):
    st.write("------> Supervisor <-----------")
    question = state["messages"][-1]

    st.write(f"Question is: {question.content}")

    template = """
    Your task is to clarify if the given user query into the following categories: [USA, Not Related].
    Only respond with the Category Name and Nothing else.

    User question: {question}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    response = chain.invoke({"question": question.content}) # Pass content of HumanMessage

    st.write(f"Parsed response from Supervisor: {response.Topic}")

    # The supervisor determines the path, but the actual source is set by RAG/LLM nodes
    return {"messages": [AIMessage(content=response.Topic)]} # Return as AIMessage for consistency

# RAG Function : Function 2
def function_2(state: AgentState):
    st.write("============> RAG Node <======================")
    question = state["messages"][0] # Original user question

    if st.session_state.retriever is None:
        st.error("No file uploaded for RAG. Please upload a text file first.")
        # Fallback or indicate an error state, perhaps route to LLM or end
        return {"messages": [AIMessage(content="I cannot answer this question using RAG as no context file has been uploaded.")], "response_source": "Error/No RAG Context"}

    prompt = PromptTemplate(
        template="""
        You are an assistant for answering the given question and provide accurate answers.
        Use the following pieces retrieved from the context to answer the question.
        If you don't know the answer, then you just say that you don't know the answer.
        Use a maximum of three sentences to answer the question.

        Question: {question}

        Context: {context}

        Answer:
        """,
        input_variables=['context', 'question']
    )

    rag_chain = (
        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = rag_chain.invoke(question.content) # Pass content of HumanMessage

    return {"messages": [AIMessage(content=result)], "response_source": "RAG / Uploaded Context"}

# LLM Call
def function_3(state: AgentState):
    st.write("===========> LLM Call <=====================")
    question = state["messages"][0] # Original user question

    complete_query = "Answer the following question with your knowledge of the real world. The question is: " + question.content

    response = model.invoke(complete_query)

    return {"messages": [AIMessage(content=response.content)], "response_source": "LLM"}

# Router function
def router(state: AgentState):
    st.write("====================> Router <================")
    # The last message from the supervisor is an AIMessage with the topic
    last_message_content = state["messages"][-1].content
    st.write(f"Message received to Router: {last_message_content}")

    if "usa" in last_message_content.lower():
        return "RAG Call"
    else:
        return "LLM Call"

# Building the Orchestration workflow
workflow = StateGraph(AgentState)

# Adding Nodes
workflow.add_node("Supervisor", function_1)
workflow.add_node("RAG", function_2)
workflow.add_node("LLM", function_3)

# Specifying the beginning node
workflow.set_entry_point("Supervisor")

# Defining the conditional Edges
workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM"
    }
)

# Adding Nodes Edge
workflow.add_edge("RAG", END)
workflow.add_edge("LLM", END)

# Compile the workflow to create the app
app = workflow.compile()

# --- Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Accept user input
user_query = st.chat_input("Ask a question about the topic...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=user_query, type="human")) # Changed 'user' to 'human'
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare initial state for the graph
    initial_state = {"messages": [HumanMessage(content=user_query)], "response_source": "N/A"}

    # Run the LangGraph app
    try:
        with st.spinner("Thinking..."):
            # The output of the graph is the final state after execution
            final_state = app.invoke(initial_state)
            ai_response_message = final_state["messages"][-1]
            response_source = final_state["response_source"]
            st.session_state.response_source = response_source

            # Add AI message to chat history
            st.session_state.messages.append(AIMessage(content=ai_response_message.content, type="ai")) # Changed 'assistant' to 'ai'

            with st.chat_message("assistant"):
                st.markdown(ai_response_message.content)
                st.caption(f"Source: {response_source}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.messages.append(AIMessage(content=f"Sorry, an error occurred: {e}", type="ai")) # Changed 'assistant' to 'ai'
        with st.chat_message("assistant"):
            st.markdown(f"Sorry, an error occurred: {e}")

# Clear chat history button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.response_source = "N/A"
    st.session_state.db = None # Also clear the DB to allow new file upload
    st.session_state.retriever = None
    st.experimental_rerun() # Rerun the app to clear display

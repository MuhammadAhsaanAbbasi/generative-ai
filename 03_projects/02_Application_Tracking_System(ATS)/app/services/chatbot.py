from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.schema import BaseMessage
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_pinecone import PineconeVectorStore
from pinecone.exceptions import PineconeException
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from db.config import DB_SESSION, ChatSession
from sqlmodel import select
import tempfile
import json
import time
import os

_ = load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

import re

# Utility Functions
def sanitize_file_name(file_name):
    """Sanitize file names for use as Pinecone index names."""
    return re.sub(r'[^a-z0-9-]', '', file_name.lower().replace(" ", "-")).strip('-')


def pdf_save(file_name, texts):
    """Save vectors to Pinecone and return index details."""
    sanitized_file_name = sanitize_file_name(file_name)

    # Check if index exists, else create it
    if sanitized_file_name not in pc.list_indexes():
        try:
            pc.create_index(
                name=sanitized_file_name,
                dimension=768,  # Correct dimension for GoogleGenerativeAIEmbeddings
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(sanitized_file_name).status["ready"]:
                time.sleep(1)
        except PineconeException as e:
            raise PineconeException(f"Error creating index: {e}")

    # Upload embeddings to Pinecone
    index = pc.Index(sanitized_file_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(texts)
    return {"index_name": sanitized_file_name}


def pdf_retriever(file_upload):
    """Process PDF, split text, and save embeddings in Pinecone."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_upload.read())
        temp_file_path = temp_file.name

    # Load & split PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    # Save to Pinecone
    file_name = file_upload.name
    index_details = pdf_save(file_name, texts)
    
    # Return both the index name and the text chunks as context
    return {"index_name": index_details["index_name"]}

def retrieve_answer(job_desc, user_query, index_name, session: DB_SESSION, chat_id: str):
    # Define the system prompt and user query template
    input_prompt = """
     You are a skilled Applicant Tracking System (ATS) with deep expertise in the field of any one job role from Data Science, Full-Stack Web Development, Big Data Engineering, DevOps, AI/ML & Data Analyst & deep ATS Functionality .
    Your task is to evaluate the Job Description against the provided Resume Context and return a response based on the user_query.

    Job Description: {job_description}
    Resume Context:
    <context>
    {context}
    </context>
    user_query: {input}
    """

    # Create the prompt and document chain for LangChain
    prompt = ChatPromptTemplate.from_template(input_prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve relevant context from Pinecone (use the index from the uploaded resume)
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create the retrieval chain (this is a tool that fetches context)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Now wrap the retrieval_chain as a tool (to be used by the AgentExecutor)
    def retrieval_tool(query: str, job_description: str):
        return retrieval_chain.invoke({"input": query, "job_description": job_description})["answer"]
    
    retrieval_tools = Tool(
        name="retrieval_tool",
        description="A tool for retrieving relevant context from Pinecone based on the user's query & job description",
        func=retrieval_tool
    )

    tools = [retrieval_tools]

    # Ensure all required variables for the prompt are included
    updated_prompt = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    agent = create_structured_chat_agent(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(updated_prompt),
        tools=tools
    )

    # Retrieve chat history from the database
    db_chat = session.exec(select(ChatSession).where(ChatSession.id == chat_id)).first()
    if db_chat:
        # Parse the chat history from the database
        try:
            chat_history_data = json.loads(db_chat.chat_messages)
        except json.JSONDecodeError:
            raise Exception("Invalid chat history format")
        
        # Convert chat history into LangChain messages
        chat_history = [
            HumanMessage(**msg) if msg["type"] == "human" else AIMessage(**msg)
            for msg in chat_history_data
        ]
        
        # Initialize LangChain memory with the previous chat history
        chat_memory = ChatMessageHistory(messages=chat_history)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=chat_memory)
    else:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Use the tool in the AgentExecutor along with memory
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Run the agent to process the user query and context
    response = agent_executor.invoke({"input": user_query, "job_description": job_desc})

    # Update chat history and save it back to the database
    new_chat_history = json.dumps([msg.model_dump() for msg in memory.chat_memory.messages])
    
    if db_chat:
        db_chat.chat_messages = new_chat_history  # Update existing chat history
    else:
        db_chat = ChatSession(chat_messages=new_chat_history, vector_index=index_name)  # Create new entry
        session.add(db_chat)
    
    session.commit()
    session.refresh(db_chat)

    return {
        "output": response["output"],
        "chat_history": db_chat.chat_messages
    }
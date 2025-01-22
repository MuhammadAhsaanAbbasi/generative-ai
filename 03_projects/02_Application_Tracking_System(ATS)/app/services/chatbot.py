from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone.exceptions import PineconeException
from services.retrieval import retrieve_tool
from db.config import DB_SESSION, ChatSession
from langchain import hub
from langchain.tools import Tool
from dotenv import load_dotenv
from pinecone import Pinecone
from sqlmodel import select
import json
import os

_ = load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def retrieve_answer(job_desc, user_query, index_name, session: DB_SESSION, chat_id: int):

    retrieval_tools = retrieve_tool(index_name=index_name)

    tools = [retrieval_tools]

    # Ensure all required variables for the prompt are included
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(  # Replaces deprecated structured agent(
        llm=llm,
        prompt=prompt,
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
        memory = ConversationBufferMemory(memory_key="chat_history",  return_messages=True)

    # Use the tool in the AgentExecutor along with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    user_query = f"{user_query} \n job_description: {job_desc}"

    # Run the agent to process the user query and context
    response = agent_executor.invoke({"input": user_query})

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
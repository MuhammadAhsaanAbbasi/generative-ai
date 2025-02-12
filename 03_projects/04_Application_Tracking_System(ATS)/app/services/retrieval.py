from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore
from pinecone.exceptions import PineconeException
from pinecone import Pinecone, ServerlessSpec
from langchain.tools import Tool
from dotenv import load_dotenv
import tempfile
import time
import os

_ = load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5)

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

def retrieve_tool(index_name):
    """Create a retrieval tool for the given index."""
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retrieval = vector_store.as_retriever(search_kwargs={"k": 3})

    # contextualize System Prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the retrieval chain of history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever=retrieval, prompt=contextualize_q_prompt
    )
    
    # Answer System Prompt
    qa_system_prompt = """
    You are a skilled Applicant Tracking System (ATS) with deep expertise in the field of any one job role from Data Science, Full-Stack Web Development, Big Data Engineering, DevOps, AI/ML, Data Analyst & Gen/Agentic AI Developer & deep ATS Functionality .
    Your task is to evaluate the Job Description against the provided Resume Context and return a response based on the user_query.

    Job Description: {job_description}
    Resume Context:
    <context>
    {context}
    </context>
    user_query: {input}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # create a stuff documents chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # create a retrieval chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    retrieval_tool = Tool(
        name="Retrieval Tool",
        func=lambda input, **kwargs: retrieval_chain.invoke(
            {"input": input, "job_description": kwargs.get("job_description", ""), "chat_history": kwargs.get("chat_history", [])}
        ),
        description="A tool for retrieving job & skills relevant context from Pinecone based on the user's query & job_description",
    )   
    return retrieval_tool
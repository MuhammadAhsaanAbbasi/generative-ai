from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
# from db.config import DB_SESSION
from pinecone import Pinecone, ServerlessSpec
import os
import time
import tempfile

_ = load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5)

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
        pc.create_index(
            name=sanitized_file_name,
            dimension=768,  # Correct dimension for GoogleGenerativeAIEmbeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(sanitized_file_name).status["ready"]:
            time.sleep(1)

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


def retrieve_answer(job_desc, user_query, index_name):
    # Define the system prompt and user query template
    input_prompt = """
     You are a skilled Applicant Tracking System (ATS) with deep expertise in the field of any one job role from Data Science, Full-Stack Web Development, Big DAta Engineering, DeVops, AI/ML & Data Analyst & deep ATS Functionality .
    Your task is to evaluate the Job Description against the provided Resume Context and return a response based on the user_query.

    Job Description: {job_description}
    Resume Context:
    Please provide the most accurate response based on the user_query
    <context>
    {context}
    <context>
    user_query: {input}
    """

    # Create the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(input_prompt)

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve relevant context from Pinecone
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # response = retriever.invoke(job_desc)

    # # Retrieve the context using the retriever (this should return the document context)
    # retrieved_context = retriever.get_relevant_documents(query=job_desc)  # Querying Pinecone for related context

    # # Combine the retrieved context with the job description for analysis
    # full_context = "\n".join([text["text"] for text in retrieved_context])

    # Execute the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": user_query, "job_description": job_desc})

    return response["answer"]

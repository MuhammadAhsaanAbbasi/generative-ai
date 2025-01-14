from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import os
import time
import tempfile

_ = load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# vector_store = PineconeVectorStore(embedding=embeddings)

import re

def sanitize_file_name(file_name):
    # Convert to lowercase, replace spaces with hyphens, and remove invalid characters
    sanitized_name = re.sub(r'[^a-z0-9-]', '', file_name.lower().replace(" ", "-"))
    # Remove consecutive hyphens
    sanitized_name = re.sub(r'-+', '-', sanitized_name)
    # Ensure it doesn't start or end with a hyphen
    sanitized_name = sanitized_name.strip('-')
    return sanitized_name


def pdf_save(file_name, texts):
    # Sanitize the file name
    sanitized_file_name = sanitize_file_name(file_name)

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Recreate index with correct dimension
    if sanitized_file_name not in existing_indexes:
        pc.create_index(
            name=sanitized_file_name,
            dimension=768,  # Correct dimension for GoogleGenerativeAIEmbeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(sanitized_file_name).status["ready"]:
            time.sleep(1)

    # Upload to Pinecone
    index = pc.Index(sanitized_file_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(texts)

    index_details = pc.describe_index(sanitized_file_name)
    return index_details



def pdf_retriever(file_upload):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_upload.read())
        temp_file_path = temp_file.name

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Use sanitized file name
    file_name = file_upload.name.lower().replace(" ", "-")
    sanitized_file_name = sanitize_file_name(file_name)

    # Upload file to Pinecone
    store = pdf_save(sanitized_file_name, texts)

    return store

    
    

st.title("Application Tracking System (A.T.S)")

chatID = st.number_input("Enter your Chat ID", step=1, key="id")

jobDesc = st.text_area("Enter your Job Description", key="desc")

file_upload = st.file_uploader("Upload your resume", type=["pdf"])

button = st.button("Submit")

if button:
    if file_upload is not None:
        file = pdf_retriever(file_upload)
        st.write(file)
    else:
        st.error("Please upload a file before clicking submit!")

# type: ignore
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import time
import os

_ = load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="deepseek-r1-distill-llama-70b"
)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(api_key=pinecone_api_key)

INDEX_NAME = "langgraphnewv2"

existing_index = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_index:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

st.title("Pinecone Chatbot with GROQ & Llama3.3")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
        """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=embedding
        st.session_state.loader=PyPDFDirectoryLoader("./books")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=PineconeVectorStore.from_documents(st.session_state.documents, st.session_state.embeddings, index_name=INDEX_NAME)



user_input = st.text_input(label="Ask a question")

if st.button("Documents Embedding"):
    with st.spinner("Processing..."):
        vector_embedding()
        st.write("Documents Embedding Completed")

if user_input:
    document_chain = create_stuff_documents_chain(llm, prompt)
    vector_store = PineconeVectorStore(index=index, embedding=embedding)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_input})
    print("Response time :",time.process_time()-start)
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
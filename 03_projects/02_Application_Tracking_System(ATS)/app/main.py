import streamlit as st
from services.chatbot import retrieve_answer
from services.retrieval import pdf_retriever
from db.config import Session, engine

st.title("Application Tracking System (A.T.S)")

file_upload = st.file_uploader("Upload Resume", type=["pdf"])
upload_file = st.button("Upload File")

if upload_file:
    if file_upload is not None:
        with st.spinner("Processing..."):
            file = pdf_retriever(file_upload)
            st.session_state.index_name = file["index_name"]
            st.write(f"Index Created: {file['index_name']}")
            st.success("File uploaded successfully!")
    else:
        st.error("Please upload a file!")

chat_id = st.number_input("Enter your Chat ID (Optional)", key="id", step=1)
job_desc = st.text_area("Enter Job Description")
user_query = st.text_input("Enter your query", key="user_query")
submit_query = st.button("Submit")

if submit_query:
    with Session(engine) as session:
        with st.spinner("Retrieving answer..."):
            response = retrieve_answer(
                chat_id=chat_id,
                job_desc=job_desc,
                user_query=user_query,
                index_name="muhammad-ahsaan-abbasi---resumepdf",
                session=session
            )
            st.write("Response:", response["output"])
            st.write("Chat History:", response["chat_history"])

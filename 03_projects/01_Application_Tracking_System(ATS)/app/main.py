import streamlit as st
from services.chatbot import pdf_retriever, retrieve_answer
from db.config import DB_SESSION

# Streamlit Frontend
st.title("Application Tracking System (A.T.S)")

chat_id = st.number_input("Enter your Chat ID (Optional)", step=1, key="id")
job_desc = st.text_area("Enter Job Description")
file_upload = st.file_uploader("Upload Resume", type=["pdf"])
user_query = st.text_input("Enter your query", key="user_query")
button = st.button("Submit")

# submit1 = st.button("Tell Me About the Resume")

# submit2 = st.button("How Can I Improvise my Skills")

# submit3 = st.button("Percentage match")
    
if button:
    if not chat_id:
        if file_upload is not None:
            file = pdf_retriever(file_upload)
            print(file["index_name"])
            response = retrieve_answer(job_desc, user_query, "muhammad-ahsaan-abbasi---resumepdf")
            st.write(response)
        else:
            st.error("Please upload a file before clicking submit!")

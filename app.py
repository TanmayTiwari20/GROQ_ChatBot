import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## Load GROQ & OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("openai_apikey")
groq_api_key = os.getenv("groq_apikey")


st.title("GROQ Chat Demo -- Llama3")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context ONLY. 
    Please provide the most accurate response based on the the question
    <context>
    {context}
    Questions : {input}
    """
)


# def clear_history():
#     if "history" in st.session_state:
#         del st.session_state["history"]


# st.session_state.uploaded_file = st.file_uploader(
#     "Upload a file:", type=["pdf", "docx", "txt"], accept_multiple_files=True
# )


def vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./files")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_document = (
            st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_document, st.session_state.embeddings
        )


prompt1 = st.text_input("Enter your questions from the document")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector store DB is ready!")


import time


if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response Time : ", time.process_time() - start)
    st.write(response["answer"])

    with st.expander("Answer fetched from here : Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------------------")

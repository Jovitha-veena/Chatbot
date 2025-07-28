# app.py
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Load Mistral locally via Ollama
llm = Ollama(model="gemma:2b")

# Setup RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("My Chatbot")

user_query = st.text_input("Ask a question about tribes in India:")

if user_query:
    result = qa_chain.run(user_query)
    st.markdown(f"**Answer:** {result}")

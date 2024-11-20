import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from bs4 import BeautifulSoup
import requests
import firebase_admin
from firebase_admin import credentials, auth
import os

# Load environment variables
load_dotenv()

# Initialize Firebase Admin SDK
cred_path = "chatweb-795fc-firebase-adminsdk-fhlxt-ca56c20602.json"  # Replace this with the actual path
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# Helper function for user authentication
def authenticate_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        # Placeholder for actual authentication logic
        return True, f"Welcome back, {user.email}!"
    except firebase_admin._auth_utils.UserNotFoundError:
        return False, "User not found. Please sign up."
    except Exception as e:
        return False, f"Error: {e}"

# Helper function for user signup
def create_user(email, password):
    try:
        user = auth.create_user(
            email=email,
            password=password
        )
        return True, "User created successfully! Please log in."
    except Exception as e:
        return False, f"Error: {e}"

# Function to create vector store from a website URL
def get_vectorstore_from_url(website_url):
    try:
        loader = WebBaseLoader(website_url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to generate a response from the vector store
def get_response(query):
    try:
        if "vector_store" not in st.session_state or not st.session_state.vector_store:
            st.error("Vector store is not initialized. Please enter a valid website URL.")
            return "Unable to retrieve response."
        retriever = st.session_state.vector_store.as_retriever()
        chat = ChatOpenAI()
        response_chain = create_retrieval_chain(
            retriever=retriever,
            retriever_prompt=ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("chat_history"),
                    HumanMessage(content=query)
                ]
            ),
            combine_docs_chain=create_stuff_documents_chain()
        )
        response = response_chain.run(chat_history=st.session_state.chat_history)
        return response
    except Exception as e:
        return f"Error generating response: {e}"

# App configuration
st.set_page_config(page_title="webQuery AI", page_icon="🤖")
st.title("Ask Me!")

# Session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login or Signup Page
if not st.session_state.logged_in:
    st.subheader("Login or Signup")
    auth_mode = st.radio("Choose mode", ["Login", "Signup"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if auth_mode == "Login":
        if st.button("Login"):
            success, message = authenticate_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.success(message)
            else:
                st.error(message)

    elif auth_mode == "Signup":
        if st.button("Signup"):
            success, message = create_user(email, password)
            if success:
                st.success(message)
            else:
                st.error(message)
else:
    # Main Chatbot Interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        website_url = st.text_input("Enter Website URL")
        if website_url:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)

    user_query = st.text_input("Type your message here...")
    if user_query and user_query.strip():
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Logout Button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.pop("chat_history", None)
        st.session_state.pop("vector_store", None)
        st.success("Logged out successfully!")

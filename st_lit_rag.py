import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rag import rag_system

load_dotenv()

st.set_page_config(page_title="Streaming bot", page_icon="?")

@st.cache_resource
def initialize_app():
    # Need the code below to execute once only during initiliazation: document loading which should only occur at the very beginning
    endpoint = "https://ragsearchone.search.windows.net"
    key = "FKJIdK6JYE8Xn3hrQfbris1YByzM9RBy9gbQ455BZ7AzSeBEu8zL"
    direc_path = "C:/Users/maxst/OneDrive/Desktop/KPMG/RAG/Shrinked"
    google_api_key = "AIzaSyAMrq3rxV8PiBNOeJlYH4QiU9Yl8HQJU3Q"

    chain = rag_system(endpoint,key,google_api_key, direc_path)
    return chain


rag_app = initialize_app()

#retriever = rag_app.embed_and_store(rag_app.splits)



st.title("Streaming Bot")




# Initialise the chat history
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []


# Conversation
for message in st.session_state.chat_history: 
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)



# Interface for user input
user_query = st.chat_input("Your message")

if user_query is not None and user_query != "": 
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        #st.markdown("I dont know")
        st.markdown(rag_app.rag_chain(rag_app.retriever, user_query))



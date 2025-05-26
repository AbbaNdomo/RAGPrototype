__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import glob
import streamlit as st
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from chromadb.config import Settings
import chromadb
import asyncio
import nest_asyncio

# Allow re-entry into the existing loop if it's been closed
nest_asyncio.apply()

# If you really want to be 100% sure we're on a fresh loop:
if asyncio.get_event_loop().is_closed():
    asyncio.set_event_loop(asyncio.new_event_loop())

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from prompt import EXAMPLE_PROMPT, PROMPT, PARSING_INSTRUCTIONS
from uuid import uuid5, NAMESPACE_DNS

load_dotenv()

# Streamlit settings
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 Document Chatbot with RAG")

st.markdown("### 📤 Upload new PDF to Knowledge Base")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    dest_path = os.path.join("fin-rag", "app", "knowledge", uploaded_file.name)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded and saved `{uploaded_file.name}` to knowledge base. Please reload to parse.")

if st.button("🔄 Reload Knowledge Base"):
    st.cache_resource.clear()
    st.experimental_rerun()

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIRECTORY = os.path.join(BASE_DIR, "knowledge")
PERSIST_DIRECTORY = os.path.join(KNOWLEDGE_DIRECTORY, "db")
@st.cache_resource
def load_documents():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KNOWLEDGE_DIRECTORY = os.path.join(BASE_DIR, "knowledge")

    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIRECTORY, '*.pdf'))

    if not pdf_files:
        st.warning("No PDF files found in the knowledge directory.")
        return []

    parser = LlamaParse(result_type="markdown", parsing_instructions=PARSING_INSTRUCTIONS)
    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_files=pdf_files, file_extractor=file_extractor
    ).load_data()

    # Filter out empty documents before proceeding
    documents = [doc for doc in documents if doc.text and doc.text.strip()]

    if not documents:
        st.error("Parsed documents are empty. Please check your PDF contents.")
        return []

    for doc in documents:
        doc.metadata['source'] = os.path.basename(doc.metadata.get("file_name", "unknown"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
    split_docs = splitter.split_documents([doc.to_langchain_format() for doc in documents])

    for i, doc in enumerate(split_docs):
        doc.metadata["source"] = f"chunk_{i}::" + doc.metadata["source"]

    st.success(f"Loaded and split {len(split_docs)} document chunks.")
    return split_docs

@st.cache_resource
def build_vector_store(_docs):
    docs = _docs  # re-bind to your old name
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import uuid

    # Filter and deduplicate docs
    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        raise ValueError("No valid documents to embed.")

    encoder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    ids = [str(uuid5(NAMESPACE_DNS, doc.page_content)) for doc in docs]
    seen_ids = set()
    unique_docs, unique_ids = [], []

    for doc, doc_id in zip(docs, ids):
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
            unique_ids.append(doc_id)

    texts = [doc.page_content for doc in unique_docs]
    metadatas = [doc.metadata for doc in unique_docs]

    # Embed safety check
    embeddings = encoder.embed_documents(texts)
    if len(embeddings) != len(texts):
        raise ValueError("Mismatch between embeddings and documents.")

    # Chroma DB
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    store = Chroma.from_texts(
        texts=texts,
        embedding=encoder,
        metadatas=metadatas,
        ids=unique_ids,
        collection_name="rag_collection",
        client=client,
        persist_directory=PERSIST_DIRECTORY,
    )

    st.success(f"Stored {len(unique_docs)} unique documents in vector DB.")
    return store

def get_llm(model_choice, temperature, streaming=False):
    if model_choice == "Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=temperature,
            # flip the flag: disable_streaming == not streaming
            disable_streaming=not streaming,
            convert_system_message_to_human=True
        )

# Sidebar UI
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Gemini", "Groq"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

# Load documents and setup chain
docs = load_documents()
store = build_vector_store(docs)
retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 5})

llm = get_llm(model_choice, temperature)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
multi_query = MultiQueryRetriever.from_llm(llm=llm, retriever=retriever, include_original=True)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=multi_query,
    memory=memory,
    chain_type_kwargs={
        "prompt": PROMPT,
        "document_prompt": EXAMPLE_PROMPT
    }
)

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Chat Input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_input)
            st.markdown(result)
            st.session_state.chat_history.append(AIMessage(content=result))

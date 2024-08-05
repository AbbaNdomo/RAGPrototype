import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from llama_parse import LlamaParse
from langchain.docstore.document import Document


# Setting up the environment

class rag_system:

    def __init__(self, doc_intelligence_endpoint, doc_intelligence_key, google_api_key, direc_path):
        self.doc_intelligence_endpoint = doc_intelligence_endpoint
        self.doc_intelligence_key = doc_intelligence_key
        #os.environ['GOOGLE_API_KEY'] = 'AIzaSyB9hPjhpqM6THjy_qn8Ne214BLL1MZpobQ'
        os.environ['GOOGLE_API_KEY'] = google_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=False)

        # Create an if condition that only executes if "db_chroma" does not exist in your directory
         # Insantiating and creating Embeddings
        model = "models/embedding-001"
        embeddings = GoogleGenerativeAIEmbeddings(model=model)

        
        if not os.path.exists("db/chroma"):
            # Loading with DocIntel
            """
            docs = []
            for file in os.listdir(direc_path):
                if file.endswith(".pdf"):
                    file_path = direc_path + "/"+ file
                    loader = AzureAIDocumentIntelligenceLoader(file_path=file_path, api_endpoint= self.doc_intelligence_endpoint, api_key=self.doc_intelligence_key, api_model="prebuilt-layout", mode="markdown", analysis_features= ["ocrHighResolution"])
                    docs += loader.load()
            
            print("Length of docs: {}".format(len(docs)))
            
            """
            # Loading with llamaParse
            def load_docs(direc_path, LLAMA_CLOUD_API_KEY):
                os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
                docs = []
                parsing_10k_instructions = '''The documents attached are various company's financial Form 10-Ks. They explore financial data, providing insight into the respective company's financial status. Answer questions using the information in these financial documents and be precise.'''
                parsed_content = {}
                def get_text_file_name(file_path):
                    return file_path.split('/')[-1].split('.')[0]
                for file in os.listdir(direc_path):
                    file_path = direc_path + "/"+ file
                    documents = LlamaParse(
                                        result_type="markdown",
                                        parsing_instructions=parsing_10k_instructions,
                                        page_separator="\n=================\n",
                                        skip_diagonal_text=True,
                                    ).load_data(file_path) # file is file directory string
                    parsed_content[file] = documents
                return parsed_content
            
            
            import nest_asyncio
            nest_asyncio.apply()

            analysis_features = ["ocrHighResolution"]

            direc_path = "/teamspace/studios/this_studio/Shrinked"
            #docs = load_docs(direc_path, endpoint, key)
            docs = load_docs(direc_path, "llx-Ideo4Vk8OrczBqU8CMwe4o5JAZtnnmw8DQZtcHUG8SxrHWb6")
            # Vector Store using ChromaDB
            vectorstore_chroma = Chroma.from_documents(splits, embeddings, persist_directory="db/chroma") #"db_chroma" is stored locally. 
            self.retriever = vectorstore_chroma.as_retriever()



            # Chunking using Langchain splitters
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            len(splits)
            self.splits = splits
        else:
            vectorstore_chroma = Chroma(persist_directory="db/chroma", embedding_function=embeddings)
            self.retriever = vectorstore_chroma.as_retriever()

        
        

    def embed_and_store(self, splits):
        # Insantiating and creating Embeddings
        model = "models/embedding-001"
        embeddings = GoogleGenerativeAIEmbeddings(model=model)

        # Vector Store using ChromaDB
        vectorstore_chroma = Chroma.from_documents(splits, embeddings)

        retriever_chroma = vectorstore_chroma.as_retriever()

        #vectorstore_chroma.add_documents(documents=splits)

        return retriever_chroma
    
    # chat_history is a list of document data types
    """
    def history_retriever(self, chat_history):
        return self.embed_and_store(chat_history)
    """

    def rag_chain(self, retriever, query):
        # Rag prompt for retrieval using LangChain Expression Language (LCEL) for simplicity

        template = """"
            You're are an assistant for question-answering tasks. Answer the following quesitons,
            if you don't know the answer, just say you don't know. Use three sentences maximum and keep the answer concise. Answer only in markdown
            format.

            Question: {question}
            Context: {context}
            Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)


        def format_docs(docs): 
            return "\n\n".join(doc.page_content for doc in docs)
        
                           
        rag_chain_chroma = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm 
            | StrOutputParser()
        )

        return rag_chain_chroma.invoke(query)


"""
def main():
    print("hello")
    endpoint = "https://docintelone.cognitiveservices.azure.com/"
    key = "64c61ce74d924ced974b3ee968e50fbe"
    direc_path = "C:/Users/maxst/OneDrive/Desktop/KPMG/RAG/Shrinked"
    google_api_key = "AIzaSyB9hPjhpqM6THjy_qn8Ne214BLL1MZpobQ"

    chain = ragChain(endpoint,key,google_api_key,direc_path)

    #splits = chain.load_docs(direc_path)
    retriever = chain.embed_and_store(chain.splits)
    h_retriever = chain.history_retriever([Document(page_content = "My name is Abba")])
    result = chain.rag_chain(retriever, "What was the total purchase price of ZeniMax Media Inc.?", h_retriever)

    print(result)

main()
"""
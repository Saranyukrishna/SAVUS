import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
st.set_page_config(page_title="SAVUS", layout="wide")
st.title("SAVUS")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

@st.cache_resource
def build_pdf_retriever(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

tools = []

if uploaded_file:
    retriever = build_pdf_retriever(uploaded_file)

    def pdf_rag_tool(query):
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])

    pdf_tool = Tool(
        name="PDF_RAG",
        func=pdf_rag_tool,
        description="Use this tool to answer questions from the uploaded PDF"
    )
    tools.append(pdf_tool)

# Ensure Tavily API key is passed explicitly
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(
    tavily_api_key=tavily_key,
    max_results=5,
    description="Search the web for real-time information"
)
tools.append(tavily_tool)

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask from PDF, web, or general knowledge...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    response = agent.run(user_input)
    st.session_state.messages.append(("assistant", response))

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)


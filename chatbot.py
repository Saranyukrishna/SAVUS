import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, initialize_agent, AgentType

st.set_page_config(page_title="SAVUS")
st.title("SAVUS - Chat with PDF & Web")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

tools = []

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def pdf_search(query):
        docs = retriever.get_relevant_documents(query)
        return "\n".join([doc.page_content for doc in docs])
    
    pdf_tool = Tool(
        name="PDF Search",
        func=pdf_search,
        description="Search information from uploaded PDF"
    )
    tools.append(pdf_tool)
    
    os.unlink(pdf_path)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

if tools:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
else:
    agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    if agent:
        with st.chat_message("assistant"):
            response = agent.run(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            st.write("Please upload a PDF first!")

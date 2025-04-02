import os
import time
import tempfile
import streamlit as st
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Setup Streamlit
st.title("Web Scraping and PDF Chatbot with Gemini")

# Initialize the vector store (Chroma or pgvector)
client = chromadb.PersistentClient(path="./vector_db")
collection_name = "content_collection"
collection = client.get_or_create_collection(collection_name)

# Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Memory to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Memory to store document embeddings
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = {}

# Function to clean and chunk the scraped or PDF text
def clean_and_chunk_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Clean text
    chunk_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return chunk_splitter.split_text(text)

# Web scraping logic
def scrape_website(url):
    # Initialize Selenium WebDriver (you can use headless mode if required)
    driver = webdriver.Edge()
    driver.get(url)
    time.sleep(3)

    # Extracting page text (You may need to extract more specific elements depending on the structure)
    page_data = driver.find_element(By.TAG_NAME, "body").text
    driver.quit()
    
    # Clean and chunk the content
    chunks = clean_and_chunk_text(page_data)

    # Generate embeddings
    embeddings = embedding_model.embed_documents(chunks)

    # Store embeddings in the vector database
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"url": url, "chunk_index": i + 1}],
            ids=[f"{url}_chunk_{i + 1}"],
            embeddings=[embeddings[i]]
        )

# Function to process uploaded PDFs
def process_pdfs(uploaded_files):
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Load PDF and extract content
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Chunk the documents
        chunks = []
        for doc in docs:
            chunks.extend(clean_and_chunk_text(doc.page_content))

        # Generate embeddings for chunks
        embeddings = embedding_model.embed_documents(chunks)

        # Store embeddings in the vector database
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"filename": uploaded_file.name, "chunk_index": i + 1}],
                ids=[f"{uploaded_file.name}_chunk_{i + 1}"],
                embeddings=[embeddings[i]]
            )

# Set up the chatbot with LLM (Gemini)
def create_chatbot():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    vectorstore = Chroma(
        persist_directory="./vector_db", 
        embedding_function=embedding_model  # Pass the embedding model object
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain

# Main user interface (web scraping or PDF upload)
option = st.radio("Choose an option", ("Scrape a Website", "Upload PDFs"))

if option == "Scrape a Website":
    url = st.text_input("Enter Website URL")
    if url:
        if st.button("Scrape and Process"):
            scrape_website(url)
            st.success(f"Website {url} scraped and processed successfully.")

elif option == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process PDFs"):
            process_pdfs(uploaded_files)
            st.success(f"{len(uploaded_files)} PDFs processed and embeddings stored.")

# Display chat interface
qa_chain = create_chatbot()
user_query = st.text_input("Ask a question:")

if user_query:
    # Query the LLM and display the answer
    response = qa_chain({"question": f'If it is not in context give as no content!\n {user_query}'})
    st.session_state.chat_history.append(("User", user_query))
    st.session_state.chat_history.append(("Bot", response["answer"]))

# Show chat history
st.subheader("Chat History")
for role, text in st.session_state.chat_history:
    st.write(f"**{role}:** {text}")

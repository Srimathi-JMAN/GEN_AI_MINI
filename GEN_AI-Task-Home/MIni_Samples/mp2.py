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
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

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

# Define the scraping function with added enhancements
def scrape_website(url):
    # Initialize Selenium WebDriver (headless mode can be used for production)
    driver = webdriver.Edge()
    driver.get(url)
    time.sleep(3)

    # To keep track of visited URLs and limit visits to 100
    visited_urls = set()
    links_to_visit = [url]
    max_links_to_visit = 100

    # Extract text and links from a page
    def extract_links_and_text(url):
        driver.get(url)
        time.sleep(2)  # Ensure the page is fully loaded

        # Get page content
        page_data = driver.find_element(By.TAG_NAME, "body").text

        # Extract all links (Anchor tags)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]

        return page_data, links

    # Function to scrape and process pages
    def scrape_page(url):
        if url not in visited_urls and len(visited_urls) < max_links_to_visit:
            print(f"Scraping: {url}")
            visited_urls.add(url)
            page_data, links = extract_links_and_text(url)

            # Clean and chunk the text from the page
            chunks = clean_and_chunk_text(page_data)

            # Generate embeddings for this page
            embeddings = embedding_model.embed_documents(chunks)

            # Store embeddings in the vector database
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"url": url, "chunk_index": i + 1}],
                    ids=[f"{url}_chunk_{i + 1}"],
                    embeddings=[embeddings[i]]
                )

            # Add new links to the queue to visit (filter out external and already visited URLs)
            for link in links:
                if urlparse(link).netloc == urlparse(url).netloc:  # Only add internal links
                    if link not in visited_urls:
                        links_to_visit.append(link)

    # Scrape pages until max_links_to_visit is reached
    while links_to_visit and len(visited_urls) < max_links_to_visit:
        current_url = links_to_visit.pop(0)
        scrape_page(current_url)
    
    driver.quit()
    st.success(f"Scraping completed. {len(visited_urls)} pages processed.")



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
    response = qa_chain({"question": user_query})
    st.session_state.chat_history.append(("User", user_query))
    st.session_state.chat_history.append(("Bot", response["answer"]))

# Show chat history
st.subheader("Chat History")
for role, text in st.session_state.chat_history:
    st.write(f"**{role}:** {text}")

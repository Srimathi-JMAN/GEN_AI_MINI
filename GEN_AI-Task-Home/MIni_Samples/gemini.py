import streamlit as st
import os
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

load_dotenv()

# Set Google API Key (Replace with your actual key)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Streamlit app title
st.title("Document Chatbot with Gemini")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=f"./vector_db")

# Create or connect to an existing Chroma collection
collection_name = "docs_collection"
collection = client.get_or_create_collection(collection_name)

# Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Session state for storing document embeddings
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = {}

# Function to process PDFs
def process_pdf(uploaded_files):
    # Clear the vector store when new files are uploaded
    # collection.delete()

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Load the PDF and extract content
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Initialize a text splitter for chunking (e.g., chunk by characters)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # Chunk the documents into smaller pieces
        chunked_docs = []
        for doc in docs:
            chunked_docs.extend(text_splitter.split_text(doc.page_content))  # Chunking the text

        # Generate embeddings (vectors) for the chunked content
        pdf_vectors = embedding_model.embed_documents(chunked_docs)

        # Store the vectors along with content and metadata in ChromaDB
        for i, chunk in enumerate(chunked_docs):
            collection.add(
                documents=[chunk],
                metadatas=[{"filename": uploaded_file.name, "chunk_index": i + 1}],
                ids=[f"{uploaded_file.name}_chunk_{i + 1}"],
                embeddings=[pdf_vectors[i]]  # Store the vector for this chunk
            )

# Function to process website URL
def process_website(url):
    # Clear the vector store when a new website is scraped
    collection.delete()

    # Fetch the website's content using requests and BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the text content
    paragraphs = soup.find_all('p')
    web_text = ' '.join([p.get_text() for p in paragraphs])

    # Chunk the website's text
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_text(web_text)

    # Generate embeddings (vectors) for the chunked content
    web_vectors = embedding_model.embed_documents(chunked_docs)

    # Store the vectors along with content and metadata in ChromaDB
    for i, chunk in enumerate(chunked_docs):
        collection.add(
            documents=[chunk],
            metadatas=[{"url": url, "chunk_index": i + 1}],
            ids=[f"{url}_chunk_{i + 1}"],
            embeddings=[web_vectors[i]]  # Store the vector for this chunk
        )

# Streamlit UI: Upload PDFs or input a website URL
upload_option = st.radio("Choose input method:", ("Upload PDFs", "Scrape Website"))

if upload_option == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        process_pdf(uploaded_files)

elif upload_option == "Scrape Website":
    website_url = st.text_input("Enter website URL:")

    if website_url:
        process_website(website_url)

# Initialize the vector store with the embedding model
vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embedding_model  # Pass the embedding model object
)

# Setup memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup the chat model (Using Gemini 2.0 Flash)
llm = genai.GenerativeModel("gemini-2.0-flash")

# Function to get top 10 related chunks based on user query
def get_top_chunks(query):
    # Embed the query
    query_embedding = embedding_model.embed_documents([query])

    # Retrieve similar chunks from the vector store
    results = vectorstore.similarity_search_with_score(query, k=10)
    top_chunks = [result[0] for result in results]

    return top_chunks

user_query = st.text_input("Ask a question:")

if user_query:
    # Get top 10 relevant chunks from the vector store
    top_chunks = get_top_chunks(user_query)

    # Display top chunks related to the query
    st.subheader("Top 10 Relevant Chunks:")
    for i, chunk in enumerate(top_chunks):
        st.write(f"{i + 1}. {chunk}")

    # Ensure content is a string and correctly formatted
    content = f"f it is not in context give as no content!\n Query: {user_query}\nContext: {''.join(top_chunks)}"

    # Generate response based on user query and context (top chunks)
    response = llm.generate_content(content)

    # Store and display the response in session state
    st.session_state.chat_history.append(("User", user_query))
    st.session_state.chat_history.append(("Bot", response.candidates[0].content.parts[0].text))

# Display chat history
st.subheader("Chat History:")
for role, text in st.session_state.chat_history:
    st.write(f"**{role}:** {text}")

import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings  # Import OpenAI embeddings correctly
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import pdfplumber
import hashlib
import urllib.parse
from dotenv import load_dotenv
import os
import psycopg2
from langchain.vectorstores.pgvector import PGVector

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize database connection (PostgreSQL with PGVector)
DATABASE_URL = "postgresql://Test_owner:npg_Uz9AJC7tZyWE@ep-dark-cell-a1cv95a1-pooler.ap-southeast-1.aws.neon.tech/Test?sslmode=require"  # Update with your actual database URL

# Function to connect to the database
def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# Function to check if a URL should be scraped (avoid unnecessary/irrelevant links)
def should_scrape(url):
    # Avoid common irrelevant links such as privacy, terms of service, etc.
    ignore_keywords = ['privacy', 'terms', 'contact', 'about', 'login', 'register', 'cookie']
    return not any(keyword in url.lower() for keyword in ignore_keywords)

# Function to scrape all links on the website
def scrape_website(url):
    scraped_text = ""
    links_to_scrape = [url]  # Start with the provided URL
    scraped_links = set()  # Keep track of scraped links
    seen_urls = set()  # Set to avoid scraping the same URL
    
    while links_to_scrape and len(scraped_links) < 100:
        current_url = links_to_scrape.pop(0)
        if current_url in seen_urls or current_url in scraped_links:
            continue  # Skip already seen or scraped URLs
        
        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs])
            scraped_text += text
            
            # Mark this URL as scraped
            scraped_links.add(current_url)
            seen_urls.add(current_url)  # Mark globally as seen

            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urllib.parse.urljoin(current_url, href)
                
                # Only scrape if the URL is within the same domain and not already seen
                if should_scrape(full_url) and full_url not in seen_urls:
                    links_to_scrape.append(full_url)

        except Exception as e:
            st.error(f"Error scraping {current_url}: {e}")
            continue
    
    return scraped_text

# Function to extract text from PDFs
def extract_pdf_text(pdf_files):
    text = ""
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to embed text into vectors using OpenAI embeddings
def embed_text_with_openai(text_chunks, connection):
    embeddings = OpenAIEmbeddings()  # Correct instantiation of OpenAIEmbeddings
    vector_store = PGVector(connection=connection)  # Correct instantiation of PGVector

    for chunk in text_chunks:
        vector = embeddings.embed(chunk)  # Get embedding for the chunk
        vector_store.add(vector)  # Add the vector to the vector store
    
    return vector_store

# Function to handle query and search for the most similar chunks in the vector store
def handle_query(query, vector_store):
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed(query)  # Embed the query
    
    # Fetch top 10 related chunks based on similarity
    results = vector_store.similarity_search(query_vector, k=10)
    return results

# Define the Streamlit UI
def main():
    st.title('Website Scraping and PDF Query System')
    
    # Choose between website link or PDF upload
    option = st.radio('Choose Input Method:', ('Enter Website URL', 'Upload PDF(s)'))
    
    # Website scraping option
    if option == 'Enter Website URL':
        url = st.text_input('Enter Website URL:')
        if st.button('Scrape Website'):
            if url:
                st.write("Scraping the website...")
                text = scrape_website(url)
                if text:
                    chunks = chunk_text(text)
                    connection = get_connection()
                    embed_text_with_openai(chunks, connection)  # Embed chunks using OpenAI
                    st.success(f"Website scraped and stored successfully! {len(chunks)} chunks.")
    
    # PDF upload option
    elif option == 'Upload PDF(s)':
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            text = extract_pdf_text(uploaded_files)
            chunks = chunk_text(text)
            connection = get_connection()
            embed_text_with_openai(chunks, connection)  # Embed chunks using OpenAI
            st.success(f"PDFs processed and stored successfully! {len(chunks)} chunks.")
    
    # Query input
    query = st.text_input('Enter your query:')
    if st.button('Send'):
        if query:
            connection = get_connection()
            vector_store = PGVector(connection=connection)  # Correct PGVector initialization
            results = handle_query(query, vector_store)
            st.write("Top 10 related chunks:")
            for idx, result in enumerate(results):
                st.write(f"{idx + 1}: {result}")

            # Call LLM with query and related chunks
            llm = OpenAI()  # You can replace this with your Gemini model
            chain = LLMChain(llm=llm)
            response = chain.run(input={"query": query, "chunks": results})
            st.session_state.chat_history.append({"query": query, "response": response})
            st.write(f"LLM Response: {response}")

if __name__ == '__main__':
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    main()

import os
import streamlit as st
import psycopg2
import requests
from bs4 import BeautifulSoup
import time
import pdfplumber
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from psycopg2.extras import Json

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize the generative model and embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

NEON_DB_URL = os.getenv("NEON_CONNECTION_STRING")
conn = psycopg2.connect(NEON_DB_URL)
cursor = conn.cursor()

cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
""")
conn.commit()

# Create table to store text and embeddings if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding VECTOR(768)  -- Adjust dimension size based on your embeddings
    );
""")
conn.commit()

# Function to extract links from the website
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.startswith("http"):
            links.add(href)
        if len(links) >= 100:  # Limit to 100 links
            break
    return links

# Function to extract text from the website
def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([para.get_text() for para in paragraphs])
    return text

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to embed text and store in Neon DB
def embed_and_store_in_db(text_chunks, batch_size=10, wait_time=20):
    vectors = []
    
    # Create embeddings in batches
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        
        try:
            # Embed the batch of documents
            embedding = embeddings.embed_documents(batch)
            
            for chunk, emb in zip(batch, embedding):
                vectors.append((chunk, emb))  # Store the chunk and its embedding
            
            # Sleep to avoid hitting rate limit if batch size is reached
            if len(batch) == batch_size:
                print(f"Batch {i // batch_size + 1} processed. Waiting for {wait_time} seconds to avoid rate limit...")
                time.sleep(wait_time)  # Throttle the requests

        except Exception as e:
            print(f"Error while embedding batch: {e}")
            time.sleep(wait_time)  # Wait before retrying in case of error
    
    # Store the embeddings in the database (assuming you're already connected to your DB)
    # Insert the text and corresponding embeddings into the table
    for text, vector in vectors:
        cursor.execute("""
            INSERT INTO embeddings (text, embedding)
            VALUES (%s, %s);
        """, (text, Json(vector)))
    conn.commit()

    st.success("Text has been embedded and stored in Neon DB.")

# Function to query Google Gemini for an answer
def query_gemini(question, context):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    
    Context:
    {context}
 
    Question: {question}
    Answer: If you can't find the answer, say that I couldn't find the answer.
    """
    response = model.generate_content(prompt)
    answer = response.text.strip() if response.text else "No response from Gemini."
    return answer

# Streamlit UI
st.title("Website Scraper and PDF Uploader")

# Radio button for selection
option = st.radio("Choose the data source", ["Scrape Website", "Upload PDFs"])

# Variables to store the current data source (website or PDFs)
current_data_source = None

if option == "Scrape Website":
    url = st.text_input("Enter website URL for scraping")
    if st.button("Scrape Website"):
        if url:
            st.write("Scraping website:", url)
            links = scrape_website(url)
            text = extract_text_from_website(url)
            text_chunks = chunk_text(text)
            embed_and_store_in_db(text_chunks)
        else:
            st.warning("Please enter a valid URL.")
    
elif option == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    if st.button("Upload PDFs"):
        if uploaded_files:
            text = extract_text_from_pdfs(uploaded_files)
            text_chunks = chunk_text(text)
            embed_and_store_in_db(text_chunks)

# Clear the database if the option changes
if 'last_option' not in st.session_state or st.session_state.last_option != option:
    st.session_state.last_option = option
    # Connect to Neon DB and clear old embeddings (if necessary)
    NEON_DB_URL = os.getenv("NEON_CONNECTION_STRING")
    conn = psycopg2.connect(NEON_DB_URL)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM embeddings;")
    conn.commit()
    cursor.close()
    conn.close()

# User query input for searching the stored context
query = st.text_input("Ask a question based on the stored data:")
if st.button("Send"):
    if query:
        query_embedding = embeddings.embed_query(query)
        # Query the table for matching embeddings (simplified search, adjust as needed)
        cursor.execute("""
            SELECT text, embedding
            FROM embeddings
            ORDER BY embedding <-> %s LIMIT %s;
        """, (Json(query_embedding), 10))  # Use the <-> operator for nearest neighbor search
        text_chunks = [row[0] for row in cursor.fetchall()]
        
        # Get the answer from Gemini model
        answer = query_gemini(query, text_chunks)
        
        # Display the answer
        st.write(f"Answer: {answer}")
        
        cursor.close()
        conn.close()
    else:
        st.warning("Please enter a question to ask.")

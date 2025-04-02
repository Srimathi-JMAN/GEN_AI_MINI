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
import os

load_dotenv()


# os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

os.environ["GOOGLE_API_KEY"] = "AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34"
genai.configure(api_key="AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# NEON_DB_URL = os.getenv("NEON_CONNECTION_STRING")
NEON_DB_URL="postgresql://Test_owner:npg_Uz9AJC7tZyWE@ep-dark-cell-a1cv95a1-pooler.ap-southeast-1.aws.neon.tech/Test?sslmode=require"
conn = psycopg2.connect(NEON_DB_URL)
cursor = conn.cursor()

cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
""")
conn.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        embedding VECTOR(768)
    );
""")
conn.commit()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join([p.get_text() for p in soup.find_all("p")])

def extract_text_from_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def update_embeddings(text_chunks):
    cursor.execute("DELETE FROM embeddings;")
    conn.commit()
    vectors = [(chunk, embeddings.embed_documents([chunk])[0]) for chunk in text_chunks]
    for text, vector in vectors:
        cursor.execute("""
            INSERT INTO embeddings (text, embedding) VALUES (%s, %s);
        """, (text, Json(vector)))
    conn.commit()
    st.success("Updated embeddings in Neon DB.")

def query_gemini(question, context):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"""
        Context:
        {context}
        
        Question: {question}
        Answer: If you can't find the answer, say that I couldn't find the answer.
    """)
    return response.text.strip() if response.text else "No response from Gemini."

st.title("Website Scraper and PDF Uploader")
option = st.radio("Choose the data source", ["Scrape Website", "Upload PDFs"])

if option == "Scrape Website":
    url = st.text_input("Enter website URL")
    if st.button("Scrape & Embed"):
        if url:
            text = scrape_website(url)
            text_chunks = chunk_text(text)
            update_embeddings(text_chunks)
        else:
            st.warning("Please enter a valid URL.")

elif option == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        current_files = [file.name for file in uploaded_files]
        previous_files = st.session_state.uploaded_files

        if set(current_files) != set(previous_files):
            text = extract_text_from_pdfs(uploaded_files)
            text_chunks = chunk_text(text)
            update_embeddings(text_chunks)
            st.session_state.uploaded_files = current_files

query = st.text_input("Ask a question:")
if st.button("Send"):
    if query:
        query_embedding = embeddings.embed_query(query)
        cursor.execute("""
            SELECT text FROM embeddings ORDER BY embedding <-> %s LIMIT 10;
        """, (Json(query_embedding),))
        text_chunks = [row[0] for row in cursor.fetchall()]
        answer = query_gemini(query, text_chunks)
        st.session_state.chat_history.append((query, answer))
        for q, a in st.session_state.chat_history:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
    else:
        st.warning("Enter a question to ask.")

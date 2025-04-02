import streamlit as st
import psycopg2
import requests
from bs4 import BeautifulSoup
import pdfplumber
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from psycopg2.extras import Json
import os
import uuid


load_dotenv()

# Set up API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Connect to Neon DB
NEON_DB_URL = "postgresql://Test_owner:npg_Uz9AJC7tZyWE@ep-dark-cell-a1cv95a1-pooler.ap-southeast-1.aws.neon.tech/Test?sslmode=require"
conn = psycopg2.connect(NEON_DB_URL)
cursor = conn.cursor()

cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
""")
conn.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        chat_id UUID NOT NULL,
        text TEXT NOT NULL,
        embedding VECTOR(768)
    );
""")
conn.commit()

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join([p.get_text() for p in soup.find_all("p")])

def extract_text_from_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def update_embeddings(chat_id, text_chunks):
    cursor.execute("DELETE FROM embeddings WHERE chat_id = %s;", (str(chat_id),))
    conn.commit()
    vectors = [(str(chat_id), chunk, embeddings.embed_documents([chunk])[0]) for chunk in text_chunks]
    for chat_id, text, vector in vectors:
        cursor.execute("""
            INSERT INTO embeddings (chat_id, text, embedding) VALUES (%s, %s, %s);
        """, (chat_id, text, Json(vector)))
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

# Sidebar for chat history
st.sidebar.title("Chat History")
if st.sidebar.button("New Chat"):
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = {"files": [], "queries": []}
    st.session_state.current_chat = new_chat_id

displayed_chat_id = None
for chat_id in st.session_state.chats.keys():
    if st.sidebar.button(f"Chat {chat_id[:8]}"):
        st.session_state.current_chat = chat_id

current_chat = st.session_state.current_chat
if current_chat is None:
    current_chat = str(uuid.uuid4())
    st.session_state.chats[current_chat] = {"files": [], "queries": []}
    st.session_state.current_chat = current_chat

chat_data = st.session_state.chats[current_chat]
st.sidebar.subheader("Uploaded Files")
st.sidebar.write(chat_data["files"])


# Main UI
st.title("Website Scraper and PDF Uploader")
option = st.sidebar.radio("Choose Data Source", ["Scrape Website", "Upload PDFs"])

if option == "Scrape Website":
    url = st.sidebar.text_input("Enter website URL")
    if st.sidebar.button("Scrape & Embed"):
        if url:
            text = scrape_website(url)
            text_chunks = chunk_text(text)
            update_embeddings(current_chat, text_chunks)
            chat_data["files"].append(url)
        else:
            st.sidebar.warning("Please enter a valid URL.")

elif option == "Upload PDFs":
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf", key=current_chat)
    if uploaded_files:
        current_files = [file.name for file in uploaded_files]
        if set(current_files) != set(chat_data["files"]):
            text = extract_text_from_pdfs(uploaded_files)
            text_chunks = chunk_text(text)
            update_embeddings(current_chat, text_chunks)
            chat_data["files"] = current_files



# query = st.text_input("Ask a question:", key="query_input")

references = []

query = st.text_input("Ask a question:")
if st.button("Send"):
    if query:
        query_embedding = embeddings.embed_query(query)
        cursor.execute("""
            SELECT text FROM embeddings WHERE chat_id = %s ORDER BY embedding <-> %s LIMIT 10;
        """, (str(current_chat), Json(query_embedding)))
        text_chunks = [row[0] for row in cursor.fetchall()]
        references = text_chunks[:10]
        answer = query_gemini(query, text_chunks)
        chat_data["queries"].append((query, answer,references))
        
        
    else:
        st.warning("Enter a question to ask.")
for entry in chat_data["queries"]:
    if len(entry) == 2:  # Old format (query, answer)
        q, a = entry
        refs = []  # No references in old data
    else:  # New format (query, answer, references)
        q, a, refs = entry
    
    st.write(f"**Q:** {q}")
    st.write(f"**A:** {a}")
    
    with st.expander("Reference Chunks"):
        for idx, ref in enumerate(refs, 1):
            st.write(f"{idx}) {ref}")


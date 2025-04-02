import streamlit as st
import psycopg2
import uuid
import json
from langchain.embeddings import OpenAIEmbeddings
from psycopg2.extras import Json

# 🔹 Database Connection
DB_URL = "postgresql://Test_owner:npg_Uz9AJC7tZyWE@ep-dark-cell-a1cv95a1-pooler.ap-southeast-1.aws.neon.tech/Test?sslmode=require"
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

# 🔹 Ensure table exists (with `chat_id`)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        chat_id TEXT NOT NULL,
        text TEXT NOT NULL,
        embedding JSONB
    );
""")
conn.commit()

# 🔹 Initialize Session State
if "chats" not in st.session_state:
    st.session_state.chats = {}

# 🔹 Function to Update Embeddings
def update_embeddings(chat_id, text_chunks):
    chat_id = str(chat_id)  # Ensure UUID is stored as string

    # Delete old embeddings for the chat
    cursor.execute("DELETE FROM embeddings WHERE chat_id = %s;", (chat_id,))
    conn.commit()

    # Compute embeddings
    embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key_here")
    vectors = [(chunk, embeddings.embed_documents([chunk])[0]) for chunk in text_chunks]

    # Insert new embeddings
    for text, vector in vectors:
        cursor.execute("""
            INSERT INTO embeddings (chat_id, text, embedding) VALUES (%s, %s, %s);
        """, (chat_id, text, Json(vector)))

    conn.commit()
    st.success("Updated embeddings in Neon DB.")

# 🔹 Sidebar for Chat History
st.sidebar.title("Chat History")
if st.sidebar.button("New Chat"):
    current_chat = uuid.uuid4()  # Generate new chat session
    st.session_state.chats[current_chat] = {"files": [], "queries": []}

# Load Existing Chats
for chat_id in st.session_state.chats.keys():
    if st.sidebar.button(f"Chat {chat_id}"):
        current_chat = chat_id

# 🔹 Chat Interface
st.title("Chat with AI")

# Handle File Upload
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)
if uploaded_files:
    text_chunks = [f.read().decode("utf-8") for f in uploaded_files]  # Extract text
    update_embeddings(current_chat, text_chunks)
    st.session_state.chats[current_chat]["files"].extend(uploaded_files)

# Handle User Queries
query = st.text_input("Ask a question:")
if query:
    st.session_state.chats[current_chat]["queries"].append(query)
    st.write(f"Response for: {query}")  # Call your AI model here

st.sidebar.write("Click on a chat to view files and queries.")


# import streamlit as st
# import psycopg2
# import requests
# from bs4 import BeautifulSoup
# import time
# import pdfplumber
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from psycopg2.extras import Json
# import os
# import uuid

# # Load environment variables
# load_dotenv()

# # API Key and DB Connection
# os.environ["GOOGLE_API_KEY"] = "AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34"
# genai.configure(api_key="AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34")

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# NEON_DB_URL = "postgresql://Test_owner:npg_Uz9AJC7tZyWE@ep-dark-cell-a1cv95a1-pooler.ap-southeast-1.aws.neon.tech/Test?sslmode=require"
# conn = psycopg2.connect(NEON_DB_URL)
# cursor = conn.cursor()

# cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS embeddings (
#         id SERIAL PRIMARY KEY,
#         chat_id UUID NOT NULL,
#         text TEXT NOT NULL,
#         embedding VECTOR(768)
#     );
# """)
# conn.commit()

# # Initialize Session State Variables
# if "chats" not in st.session_state:
#     st.session_state.chats = {}  # Stores chat metadata (files, queries)

# if "current_chat" not in st.session_state:
#     first_chat_id = str(uuid.uuid4())
#     st.session_state.chats[first_chat_id] = {"files": [], "queries": []}
#     st.session_state.current_chat = first_chat_id  # Set first chat as default

# # Get Current Chat
# current_chat = st.session_state.current_chat

# # Ensure the chat ID exists in session state
# if current_chat not in st.session_state.chats:
#     st.session_state.chats[current_chat] = {"files": [], "queries": []}

# # Sidebar for Chats
# with st.sidebar:
#     st.title("Chat History")
#     for chat_id in st.session_state.chats.keys():
#         if st.button(f"Chat {str(current_chat)[:8]}"):  # Show short UUID
#             st.session_state.current_chat = chat_id
#             st.rerun()

#     if st.button("➕ New Chat"):
#         new_chat_id = str(uuid.uuid4())
#         st.session_state.chats[new_chat_id] = {"files": [], "queries": []}
#         st.session_state.current_chat = new_chat_id
#         st.rerun()

# # Scraping & PDF Processing
# def scrape_website(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     return " ".join([p.get_text() for p in soup.find_all("p")])

# def extract_text_from_pdfs(uploaded_files):
#     text = ""
#     for uploaded_file in uploaded_files:
#         with pdfplumber.open(uploaded_file) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#     return text

# def chunk_text(text, chunk_size=500):
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# def update_embeddings(chat_id, text_chunks):
#     cursor.execute("DELETE FROM embeddings WHERE chat_id = %s;", (str(chat_id),))
#     conn.commit()
#     vectors = [(chat_id, chunk, embeddings.embed_documents([chunk])[0]) for chunk in text_chunks]
#     for chat_id, text, vector in vectors:
#         cursor.execute("""
#             INSERT INTO embeddings (chat_id, text, embedding) VALUES (%s, %s, %s);
#         """, (chat_id, text, Json(vector)))
#     conn.commit()
#     st.success(f"Embeddings updated for Chat {chat_id[:8]}.")

# # Gemini Query Processing
# def query_gemini(question, context):
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(f"""
#         Context:
#         {context}
        
#         Question: {question}
#         Answer: If you can't find the answer, say that I couldn't find the answer.
#     """)
#     return response.text.strip() if response.text else "No response from Gemini."

# # Left Sidebar: Upload PDFs or Scrape Website
# with st.sidebar:
#     st.subheader("Upload PDFs / Scrape Website")

#     option = st.radio("Choose data source", ["Upload PDFs", "Scrape Website"], key="data_source")

#     if option == "Scrape Website":
#         url = st.text_input("Enter website URL")
#         if st.button("Scrape & Embed"):
#             if url:
#                 text = scrape_website(url)
#                 text_chunks = chunk_text(text)
#                 update_embeddings(current_chat, text_chunks)
#                 st.session_state.chats[current_chat]["files"] = [url]  # Store URL in chat history
#             else:
#                 st.warning("Please enter a valid URL.")

#     elif option == "Upload PDFs":
#         uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
#         if uploaded_files:
#             file_names = [file.name for file in uploaded_files]
#             if set(file_names) != set(st.session_state.chats[current_chat]["files"]):  
#                 text = extract_text_from_pdfs(uploaded_files)
#                 text_chunks = chunk_text(text)
#                 update_embeddings(current_chat, text_chunks)
#                 st.session_state.chats[current_chat]["files"] = file_names  # Store filenames

# # Center Chat Area
# st.title(f"Chat {str(chat_id)[:8]}")

# if st.session_state.chats[current_chat]["files"]:
#     st.subheader("Files Used:")
#     for file in st.session_state.chats[current_chat]["files"]:
#         st.write(f"- {file}")

# query = st.text_input("Ask a question:")
# if st.button("Send"):
#     if query:
#         query_embedding = embeddings.embed_query(query)
#         cursor.execute("""
#             SELECT text FROM embeddings WHERE chat_id = %s ORDER BY embedding <-> %s LIMIT 10;
#         """, (current_chat, Json(query_embedding)))
#         text_chunks = [row[0] for row in cursor.fetchall()]
#         answer = query_gemini(query, text_chunks)

#         # Store query and answer in chat history
#         st.session_state.chats[current_chat]["queries"].append((query, answer))

#         st.subheader("Current Chat")
#         st.write(f"**Q:** {query}")
#         st.write(f"**A:** {answer}")

#     else:
#         st.warning("Enter a question to ask.")

# # Display previous chat history
# if st.session_state.chats[current_chat]["queries"]:
#     st.subheader("Previous Queries")
#     for q, a in st.session_state.chats[current_chat]["queries"]:
#         st.write(f"**Q:** {q}")
#         st.write(f"**A:** {a}")

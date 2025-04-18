import os
import openai
import faiss
import numpy as np
import pickle
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request    
from pydantic import BaseModel
from typing import Dict, List
from uuid import uuid4

# Config
TXT_FOLDER = "data"
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4"

INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# FastAPI app
app = FastAPI()

# In-memory session storage
session_histories: Dict[str, List[Dict]] = {}

# Models
class ChatRequest(BaseModel):
    session_id: str
    question: str

# Step 1: Load all .txt files
def load_txts(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

# Step 2: Simple text splitter
def split_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Step 3: Embed chunks
def embed_text(text):
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding

# Step 4: Build FAISS index
def build_vector_store(chunks):
    embeddings = [embed_text(c) for c in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def save_vector_store(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_store():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Step 5: Retrieve top k relevant chunks
def retrieve_similar_chunks(query, index, chunks, k=3):
    query_vec = embed_text(query)
    D, I = index.search(np.array([query_vec]).astype("float32"), k)
    return [chunks[i] for i in I[0]]

# Step 6: Generate GPT answer
def answer_query(session_id: str, query: str, index, chunks):
    top_chunks = retrieve_similar_chunks(query, index, chunks)
    context = "\n---\n".join(top_chunks)

    if session_id not in session_histories:
        session_histories[session_id] = []

    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    # Append previous messages for session continuity
    messages = session_histories[session_id] + messages

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages
    )


    reply = response.choices[0].message.content

    # Save conversation history
    session_histories[session_id].extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": reply}
    ])
    print(f"Session {session_id} history: {session_histories[session_id]}")
    # Limit history to last 10 interactions
    if len(session_histories[session_id]) > 20:
        session_histories[session_id] = session_histories[session_id][-20:]

    return reply

# Load or build the vector store
print("🔄 Initializing vector store...")
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    vector_index, stored_chunks = load_vector_store()
    print("✅ Loaded stored vector index and chunks.")
else:
    raw_texts = load_txts(TXT_FOLDER)
    all_chunks = []
    for txt in raw_texts:
        all_chunks.extend(split_text(txt))
    vector_index, stored_chunks = build_vector_store(all_chunks)
    save_vector_store(vector_index, stored_chunks)
    print(f"✅ Built and saved vector store with {len(stored_chunks)} chunks.")

# API Route
@app.post("/chat")
def chat(request: ChatRequest):
    answer = answer_query(request.session_id, request.question, vector_index, stored_chunks)
    return {"answer": answer}

@app.get("/new_session")
def new_session():
    session_id = str(uuid4())
    session_histories[session_id] = []  # Initialize empty history
    return JSONResponse(content={"session_id": session_id})

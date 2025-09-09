from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

from fastapi import UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# # Create an instance of the FastAPI class
app = FastAPI()


origins = [
    "http://localhost:3000",  # The address of our Next.js frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body model for our endpoint
class PromptRequest(BaseModel):
    prompt: str

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Living Brain API"}

# Define the AI prompt endpoint
@app.post("/api/prompt")
async def generate_response(request: PromptRequest):
    # The URL for the Ollama API
    ollama_url = "http://localhost:11434/api/generate"

    # The data payload to send to Ollama
    payload = {
        "model": "llama3:8b",
        "prompt": request.prompt,
        "stream": False  # We'll get the full response at once
    }

    try:
        # Make the POST request to the Ollama server
        response = requests.post(ollama_url, json=payload)
        
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the JSON response from Ollama
        data = response.json()
        
        # Return the 'response' field from the Ollama output
        return {"response": data.get("response")}

    except requests.exceptions.RequestException as e:
        # Handle connection errors or other request issues
        return {"error": f"Failed to connect to Ollama: {e}"}

# Ingest Document
@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    ext = Path(file.filename).suffix.lower()
    print(f"Ingesting document: {file.filename}")

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return {"status": "error", "message": f"Unsupported file type: {ext}"}

    # 1. Load the document
    # loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. Create embeddings and store in ChromaDB
    # This will create a directory called "chroma_db" to store the vectors
    embeddings = OllamaEmbeddings(model="llama3:8b")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Successfully ingested {len(splits)} chunks.")

    return {"status": "success", "filename": file.filename, "chunks": len(splits)}
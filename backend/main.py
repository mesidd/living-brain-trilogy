from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

from fastapi import UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Teaching the AI to Use Its Memory (RAG)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_ollama import OllamaEmbeddings, OllamaLLM


from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv

from fastapi import Form

load_dotenv()


# Create an instance of the FastAPI class
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

# Neo4j Connection Details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")


graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD
)

# Define the request body model for our endpoint
class PromptRequest(BaseModel):
    prompt: str

class SynthesizeRequest(BaseModel):
    topic: str

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Living Brain API"}


@app.post("/api/prompt")
async def generate_response(request: PromptRequest):
    # --- Vector RAG (what we had before) ---
    embeddings = OllamaEmbeddings(model="llama3:8b")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    
    # --- Graph RAG ---
    # We will use the graph to answer questions about relationships
    llm = OllamaLLM(model="llama3:8b")
    
    # This chain will specifically query the knowledge graph
    graph_chain = GraphCypherQAChain.from_llm(
        graph=graph, 
        llm=llm, 
        allow_dangerous_requests=True,
        verbose=True # Set to True to see the generated Cypher queries in your terminal
    )

    # --- The Decider ---
    # We'll use a simple logic: if the question is about relationships, use the graph.
    # Otherwise, use the vector store. A more advanced system could use an LLM to decide.
    relationship_keywords = ["relationship", "connect", "link", "between", "how does"]
    
    if any(keyword in request.prompt.lower() for keyword in relationship_keywords):
        print("--- Using Knowledge Graph to answer ---")
        # Use the GraphCypherQAChain for questions about relationships
        result = graph_chain.invoke({"query": request.prompt})
        return {"response": result.get("result")}
    else:
        print("--- Using Vector Store to answer ---")
        # Use the standard RAG chain for other questions
        system_prompt = (
            "You are an intelligent assistant. Use the following context to "
            "answer the user's question. If you don't know the answer, say you "
            "don't know."
            "\n\n"
            "{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = retrieval_chain.invoke({"input": request.prompt})
        return {"response": response.get("answer")}

# Ingest Document
@app.post("/api/ingest")
async def ingest_document(label: str = Form(...),
    file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    ext = Path(file.filename).suffix.lower()
    print(f"Ingesting document: {file.filename}")

    # 1. Load the document
    loader = PyPDFLoader(file_path) if file.filename.endswith(".pdf") else TextLoader(file_path)

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
    print(f"Successfully ingested {len(splits)} chunks into ChromaDB.")
    # print(f"Successfully ingested {len(splits)} chunks.")

    # --- NEW: Graph Extraction Section ---
    print(f"Extracting entities and relationships for the {label} graph...")

    # We will use a specific LLM for graph extraction
    graph_extraction_llm = OllamaLLM(model="llama3:8b")

   # A prompt to instruct the LLM to extract triples
    extraction_prompt = PromptTemplate.from_template(
        "You are an expert at extracting information. From the following text, "
        "extract entities and relationships as a list of triples. "
        "Format each triple as (Head, RELATION, Tail). "
        "Example: (Marie Curie, FOUND, Radium). "
        "Do not add any explanation or preamble. Just provide the list of triples.\n\n"
        "Text: '''{text}'''"
    )

    # We'll process the document in larger chunks for better context
    combined_text = " ".join([doc.page_content for doc in documents])

    # Generate the triples
    formatted_prompt = extraction_prompt.format(text=combined_text)
    extracted_triples_str = await graph_extraction_llm.ainvoke(formatted_prompt)

    # Parse the string response into a list of tuples
    # This is a simple parser, can be improved for robustness
    triples = []

    for chunk in splits:
        prompt_text = extraction_prompt.format(text=chunk.page_content)
        chunk_triples_str = await graph_extraction_llm.ainvoke(prompt_text)
        
        # Original
        for line in extracted_triples_str.strip().split('\n'):
            try:
                head, rel, tail = line.strip().strip('()').split(', ')
                triples.append((head.strip(), rel.strip(), tail.strip()))
            except ValueError:
                continue # Skip malformed lines

    # Add to Neo4j graph
    for head, rel, tail in triples:
        graph.query(
            "MERGE (h:`Entity`:`" + label.capitalize() + "` {name: $head}) "
            "MERGE (t:`Entity`:`" + label.capitalize() + "` {name: $tail}) "
            "MERGE (h)-[:`" + rel.replace(" ", "_").upper() + "`]->(t)",
            params={'head': head, 'tail': tail}
        )
    
    print(f"Successfully added {len(triples)} relationships to the knowledge graph under the {label}.")

    return {"status": "success", "filename": file.filename, "label": label, "chunks": len(splits), "triples": len(triples)}


@app.post("/api/synthesize")
async def synthesize_topics(request: SynthesizeRequest):
    print(f"--- Starting synthesis for topic: {request.topic} ---")

    # 1. Pull ALL knowledge from both sides of the graph
    # This is more robust as it doesn't depend on a topic keyword match.
    
    thesis_query = "MATCH (n:Thesis)-[r]->(m:Thesis) RETURN n.name AS head, type(r) AS relation, m.name AS tail LIMIT 25"
    thesis_results = graph.query(thesis_query)
    thesis_triples = [f"({r['head']}, {r['relation']}, {r['tail']})" for r in thesis_results]
    thesis_context = "\n".join(thesis_triples)

    antithesis_query = "MATCH (n:Antithesis)-[r]->(m:Antithesis) RETURN n.name AS head, type(r) AS relation, m.name AS tail LIMIT 25"
    antithesis_results = graph.query(antithesis_query)
    antithesis_triples = [f"({r['head']}, {r['relation']}, {r['tail']})" for r in antithesis_results]
    antithesis_context = "\n".join(antithesis_triples)

    print(f"Found {len(thesis_triples)} thesis points and {len(antithesis_triples)} antithesis points.")
    
    # Check if we actually found any data
    if not thesis_context or not antithesis_context:
        return {"error": "Could not find sufficient thesis or antithesis data in the graph to perform synthesis."}

    # 2. First LLM Call: Identify the core conflict (this part is the same)
    conflict_prompt = PromptTemplate.from_template(
        "You are a master debater and analyst. Below are two opposing sets of ideas, a Thesis and an Antithesis, presented as knowledge graph triples.\n\n"
        "Thesis:\n{thesis_context}\n\n"
        "Antithesis:\n{antithesis_context}\n\n"
        "Your task is to analyze both viewpoints and identify the 2-3 core points of tension or fundamental disagreement between them. Summarize this conflict clearly and concisely."
    )
    llm = OllamaLLM(model="llama3:8b")
    conflict_chain = conflict_prompt | llm
    print("--- Identifying conflict... ---")
    conflict = await conflict_chain.ainvoke({
        "thesis_context": thesis_context, 
        "antithesis_context": antithesis_context
    })
    print(f"Conflict identified: {conflict}")

    # 3. Second LLM Call: Generate the synthesis (this part is the same)
    synthesis_prompt = PromptTemplate.from_template(
        "You are a visionary philosopher. You have been presented with two opposing viewpoints (a Thesis and an Antithesis) and an analysis of their core conflict.\n\n"
        "Thesis triples:\n{thesis_context}\n\n"
        "Antithesis triples:\n{antithesis_context}\n\n"
        "Core Conflict:\n{conflict}\n\n"
        "Your task is to generate a 'Synthesis' - a new, higher-level perspective that resolves, integrates, or transcends this conflict. "
        "The synthesis should not simply pick a side, but create a more nuanced understanding of the topic: '{topic}'."
    )
    synthesis_chain = synthesis_prompt | llm
    print("--- Generating synthesis... ---")
    synthesis = await synthesis_chain.ainvoke({
        "thesis_context": thesis_context,
        "antithesis_context": antithesis_context,
        "conflict": conflict,
        "topic": request.topic
    })
    print(f"Synthesis generated: {synthesis}")

    return {"thesis": thesis_context, "antithesis": antithesis_context, "conflict": conflict, "synthesis": synthesis}
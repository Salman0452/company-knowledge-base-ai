import os
import shutil
import tempfile
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()

# ── APP SETUP ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Company Knowledge Base API",
    description="RAG-powered document Q&A system",
    version="1.0.0"
)

# Allow Streamlit (running on port 8501) to talk to FastAPI (running on port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── SHARED RESOURCES ───────────────────────────────────────────────────────────
# These are initialized once and reused across all requests (performance)
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY
)

# In-memory conversation store per session
# Key = session_id (sent from Streamlit), Value = memory object
memory_store: dict[str, ConversationBufferWindowMemory] = {}

def get_vectorstore() -> Chroma:
    """Load existing ChromaDB. Raises error if not initialized yet."""
    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload a PDF first."
        )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create memory for a given session."""
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return memory_store[session_id]


# ── REQUEST/RESPONSE MODELS ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"          # Streamlit sends a unique ID per user
    selected_docs: list[str] = []        # Optional document filter

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]                  # [{doc_name, page, preview}]

class UploadResponse(BaseModel):
    message: str
    doc_name: str
    chunks_added: int
    total_docs: list[str]


# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms API is running."""
    return {"status": "running", "app": "Company Knowledge Base API"}

@app.get("/documents")
def list_documents():
    """
    Returns list of all document names currently in ChromaDB.
    Streamlit sidebar calls this to populate the filter.
    """
    try:
        vectorstore = get_vectorstore()
        all_meta = vectorstore.get()["metadatas"]
        doc_names = sorted(set(
            m["doc_name"] for m in all_meta if "doc_name" in m
        ))
        return {"documents": doc_names, "count": len(doc_names)}
    except HTTPException:
        # No DB yet — return empty list instead of crashing
        return {"documents": [], "count": 0}

@app.get("/documents")
def list_documents():
    """
    Returns list of all document names currently in ChromaDB.
    Streamlit sidebar calls this to populate the filter.
    """
    try:
        vectorstore = get_vectorstore()
        all_meta = vectorstore.get()["metadatas"]
        doc_names = sorted(set(
            m["doc_name"] for m in all_meta if "doc_name" in m
        ))
        return {"documents": doc_names, "count": len(doc_names)}
    except HTTPException:
        # No DB yet — return empty list instead of crashing
        return {"documents": [], "count": 0}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, chunks it, embeds it, and stores in ChromaDB.
    Does NOT wipe existing data — it ADDS to the existing vector store.
    This means users can upload multiple PDFs one by one.
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temp location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load and chunk the PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )
        chunks = splitter.split_documents(pages)

        # Tag chunks with metadata (same logic as your original ingest.py)
        doc_name = file.filename.replace(".pdf", "").replace("-", " ").replace("_", " ")
        for chunk in chunks:
            chunk.metadata["doc_name"] = doc_name
            chunk.metadata["filename"] = file.filename

        # Add to ChromaDB — this APPENDS, doesn't overwrite
        # So existing documents stay safe when a new one is uploaded
        if os.path.exists(CHROMA_DIR):
            # DB exists — load it and add new chunks
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
        else:
            # First upload — create fresh DB
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )

        # Get updated doc list to return to frontend
        vectorstore = get_vectorstore()
        all_meta = vectorstore.get()["metadatas"]
        all_docs = sorted(set(m["doc_name"] for m in all_meta if "doc_name" in m))

        return UploadResponse(
            message=f"Successfully ingested '{file.filename}'",
            doc_name=doc_name,
            chunks_added=len(chunks),
            total_docs=all_docs
        )

    finally:
        # Always clean up the temp file — no leftover files on server
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accepts a question + optional doc filter + session_id.
    Returns the LLM answer + source document snippets.
    """
    vectorstore = get_vectorstore()

    # Build retriever — apply document filter if specific docs selected
    all_meta = vectorstore.get()["metadatas"]
    all_doc_names = [m["doc_name"] for m in all_meta if "doc_name" in m]

    if request.selected_docs and len(request.selected_docs) < len(set(all_doc_names)):
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "filter": {"doc_name": {"$in": request.selected_docs}}
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    # Get session memory (keeps conversation history per user)
    memory = get_memory(request.session_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    # System prompt injected into question — same approach as your original app
    augmented_prompt = f"""You are a company policy assistant.
Answer based ONLY on the provided context.
If the answer is not in the context, say: 'I could not find this in the selected documents.'
Do not make up information.

Question: {request.question}"""

    result = chain.invoke({"question": augmented_prompt})
    answer = result["answer"]
    source_docs = result["source_documents"]

    # Format sources cleanly for the frontend
    sources = [
        {
            "doc_name": doc.metadata.get("doc_name", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "preview": doc.page_content[:200] + "..."
        }
        for doc in source_docs
    ]

    return QueryResponse(answer=answer, sources=sources)

@app.delete("/documents/{doc_name}")
def delete_document(doc_name: str):
    """
    Deletes all chunks belonging to a specific document from ChromaDB.
    Useful when user wants to remove a document and re-upload a newer version.
    """
    vectorstore = get_vectorstore()
    
    # Get IDs of all chunks matching this doc_name
    results = vectorstore.get(where={"doc_name": doc_name})
    ids_to_delete = results["ids"]

    if not ids_to_delete:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")

    vectorstore.delete(ids=ids_to_delete)

    return {
        "message": f"Deleted '{doc_name}' ({len(ids_to_delete)} chunks removed)",
        "deleted_chunks": len(ids_to_delete)
    }



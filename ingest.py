import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── 1. LOAD DOCUMENTS ──────────────────────────────────────────────────────────
# DirectoryLoader loads ALL PDFs in the data/ folder automatically
loader = DirectoryLoader(
    path="data/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

print("Loading documents...")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ── 2. CHUNK THE DOCUMENTS ─────────────────────────────────────────────────────
# You already understand chunking from Week 1.
# RecursiveCharacterTextSplitter tries to split on paragraphs first,
# then sentences, then words — it's smarter than CharacterTextSplitter.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # characters per chunk
    chunk_overlap=50,      # overlap so context isn't lost at boundaries
    separators=["\n\n", "\n", " ", ""]  # tries these in order
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ── 3. CREATE EMBEDDINGS ───────────────────────────────────────────────────────
# This model runs locally — no API key, no cost, no rate limits.
# First run will download ~90MB model. Subsequent runs use cache.
print("Loading embedding model (first run downloads ~90MB)...")
embedding_model = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

# ── 4. STORE IN CHROMADB ───────────────────────────────────────────────────────
# Chroma.from_documents does THREE things at once:
#   a) converts each chunk to a vector using embedding_model
#   b) stores the vector in ChromaDB
#   c) stores the original text alongside the vector (for retrieval)
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"   # saves to disk
)

print(f"Done! {len(chunks)} chunks embedded and stored in chroma_db/")
print("You can now run app.py to start querying.")
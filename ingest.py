import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

load_dotenv()



# ── 1. LOAD ALL PDFs MANUALLY WITH METADATA ────────────────────────────────────
# We load each PDF individually so we can tag each chunk with its document name
# This is better than DirectoryLoader when you need metadata control

DATA_DIR = "data/"
all_chunks = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 75
)
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Loading {filename}...")

        loader = PyPDFLoader(filepath)
        pages = loader.load()

        chunks = splitter.split_documents(pages)

        # Tag each chunk with clean document name — this enables filtering later
        doc_name = filename.replace(".pdf", "").replace("-", " ").replace("_", " ")
        for chunk in chunks:
            chunk.metadata["doc_name"] = doc_name
            chunk.metadata["filename"] = filename
        
        print(f"-> {len(chunks)} chunks")
        all_chunks.extend(chunks)

print(f"\n Totall chunks across all documents: {len(all_chunks)}")

# ── 2. EMBED AND STORE ─────────────────────────────────────────────────────────
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

# Delete old chroma_db and rebuild fresh
import shutil
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("Cleared old vector store")

vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


print(f"Stored in ChromaDB successfully")
print(f"Documents ingested: {set([c.metadata['doc_name'] for c in all_chunks])}")
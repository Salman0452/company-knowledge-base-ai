# Company Knowledge Base AI

> Chat with your company documents using natural language — powered by 
> RAG, semantic search, and Groq LLaMA 3.3 70B.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://company-knowledge-base-ai-8qx94fsufxkhehkmmi9dka.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)

## Live Demo
[Try it here](https://company-knowledge-base-ai-8qx94fsufxkhehkmmi9dka.streamlit.app/)

---

## The Problem It Solves

Traditional keyword search fails when users don't use exact words 
from the document. This app uses vector embeddings to understand 
**meaning**, not just keywords.

Example:
- User asks: *"Can I work from another city?"*
- Document says: *"Employee relocation policy..."*
- Keyword search: No match
- This app: Finds it instantly

---

## Architecture
```
PDF Documents
     ↓
LangChain PDF Loader
     ↓
RecursiveCharacterTextSplitter (chunk_size=500, overlap=75)
     ↓
Cohere Embeddings (embed-english-v3.0)
     ↓
ChromaDB Vector Store (persisted on disk)
     ↓
User Query → Cohere Embeddings → Similarity Search (top 4 chunks)
     ↓
Groq LLaMA 3.3 70B → Answer with Sources
```

---

## Features

- **Semantic Search** — finds meaning, not just keywords
- **Multi-Document Support** — ingest unlimited PDFs
- **Metadata Filtering** — search specific documents only
- **Source Transparency** — every answer shows exactly which 
  document chunks were used
- **Hallucination Prevention** — says "I don't know" when 
  answer isn't in documents
- **Conversation Memory** — remembers last 3 turns
- **100% Free APIs** — Groq + Cohere + ChromaDB

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq API — LLaMA 3.3 70B |
| Embeddings | Cohere embed-english-v3.0 |
| Vector DB | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Project Structure
```
company-knowledge-base-ai/
│
├── ingest.py        # Document loading, chunking, embedding pipeline
├── retriever.py     # Vector similarity search and testing
├── app.py           # Streamlit UI + RAG chain
│
├── data/            # Place your PDF documents here
├── chroma_db/       # Auto-generated vector store
│
├── requirements.txt
├── .env.example     # Template for API keys
└── README.md
```

---

## Run Locally
```bash
git clone https://github.com/yourusername/company-knowledge-base-ai
cd company-knowledge-base-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys:
```
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```
```bash
# Add your PDFs to data/ folder, then:
python ingest.py      # build vector store
streamlit run app.py  # launch app
```

---

## Key Learning — Why RAG?

Large language models have a knowledge cutoff and no access to 
your private documents. RAG solves this by:

1. Converting your documents into searchable vectors
2. Finding relevant chunks at query time
3. Passing only those chunks to the LLM as context
4. Getting a grounded answer — not a hallucination

---

## What I'd Add Next

- [ ] Pinecone for hosted vector storage
- [ ] Document upload directly from UI
- [ ] Answer confidence scores
- [ ] Multi-language support

---

## Author

Built as part of a 30-day AI Engineer bootcamp.
Connect with me on [LinkedIn](inkedin.com/in/salman-ahmad-dev/)
```

---

## Step 3 — Add `.env.example`

Create this file so others can run your project:
```
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
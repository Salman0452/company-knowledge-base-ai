# Company Knowledge Base AI

> A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload any PDF and query it using natural language — powered by FastAPI, ChromaDB, Cohere Embeddings, and Groq LLM.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-orange)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple)

## Live Demo

| Service | URL |
|---|---|
| Frontend (Streamlit) | https://frontend-production-b904.up.railway.app |
| Backend API (FastAPI) | https://company-knowledge-base-ai-production.up.railway.app |
| API Documentation | https://company-knowledge-base-ai-production.up.railway.app/docs |

---

## What This Does

Upload any PDF document and immediately ask questions about it in natural language. The system retrieves the most relevant chunks from your documents and uses an LLM to generate accurate, grounded answers — with source citations so you can verify every response.

**Real-world use cases:**
- Company policy Q&A for HR teams
- Legal document analysis
- Research paper querying
- Technical documentation assistant

---

## Architecture

```
User uploads PDF
      ↓
FastAPI /upload endpoint
      ↓
PDF → chunks (500 tokens) → Cohere embeddings → ChromaDB
      ↓
User asks question via Streamlit
      ↓
FastAPI /query endpoint
      ↓
ChromaDB similarity search → Top 4 relevant chunks
      ↓
Groq LLaMA 3.3 70B generates answer with sources
      ↓
Response displayed with page citations
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Chat interface + PDF upload UI |
| Backend | FastAPI | REST API, business logic |
| LLM | Groq (LLaMA 3.3 70B) | Answer generation |
| Embeddings | Cohere (embed-english-v3.0) | Semantic search |
| Vector DB | ChromaDB | Document storage & retrieval |
| Orchestration | LangChain | RAG pipeline |
| Container | Docker + docker-compose | Deployment |

---

## Quick Start

### Prerequisites
- Docker & docker-compose installed
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Cohere API key (free at [cohere.com](https://cohere.com))

### 1. Clone the repo
```bash
git clone https://github.com/Salman0452/company-knowledge-base-ai.git
cd company-knowledge-base-ai
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Open the app
- **Streamlit UI:** http://localhost:8501
- **FastAPI docs:** http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/documents` | List all ingested documents |
| POST | `/upload` | Upload and ingest a PDF |
| POST | `/query` | Query documents with natural language |
| DELETE | `/documents/{doc_name}` | Remove a document |

### Example query request
```json
POST /query
{
  "question": "What is the remote work policy?",
  "session_id": "user-123",
  "selected_docs": ["company handbook"]
}
```

### Example response
```json
{
  "answer": "According to the company handbook, employees may work remotely up to 3 days per week...",
  "sources": [
    {
      "doc_name": "company handbook",
      "page": 12,
      "preview": "Remote work policy: Employees are eligible..."
    }
  ]
}
```

---

## Features

- **Dynamic PDF upload** — no hardcoded documents, upload anything at runtime
- **Multi-document support** — ingest multiple PDFs and filter by document
- **Conversation memory** — remembers last 3 exchanges per session
- **Source citations** — every answer shows which page it came from
- **Document management** — upload and delete documents via UI
- **Production-ready** — fully containerized with Docker

---

## Project Structure

```
company-knowledge-base-ai/
├── backend/
│   └── main.py          # FastAPI app — all endpoints
├── frontend/
│   └── app.py           # Streamlit UI
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service orchestration
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variable template
```

---

## Key Technical Decisions

**Why ChromaDB over Pinecone?**
ChromaDB runs locally with zero setup — perfect for self-hosted deployments. No external API calls for vector storage means lower latency and no additional cost.

**Why Groq over OpenAI?**
Groq's hardware delivers significantly faster inference on open-source models like LLaMA 3.3 70B. Free tier is generous enough for production demos.

**Why separate FastAPI + Streamlit instead of one Streamlit app?**
Separation of concerns — the FastAPI backend can serve any frontend (mobile app, web app, other services). Streamlit is just one client.

---

## Author

**Salman Ahmad** — AI Engineer
- GitHub: [@Salman0452](https://github.com/Salman0452)

---

## License

MIT License — feel free to use this for your own projects.
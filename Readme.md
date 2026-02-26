[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://company-knowledge-base-ai-8qx94fsufxkhehkmmi9dka.streamlit.app/)

## Live Demo
[Try it here](https://company-knowledge-base-ai-8qx94fsufxkhehkmmi9dka.streamlit.app/)

# Company Knowledge Base AI

A RAG (Retrieval-Augmented Generation) application that lets you chat with 
your company's HR documents using natural language.

## Architecture

PDF Documents → LangChain Loader → Text Chunker → Cohere Embeddings 
→ ChromaDB Vector Store → Groq LLaMA 3.3 70B → Streamlit UI

## Features

- Semantic search (finds meaning, not just keywords)
- Source transparency (shows exactly which document chunks were used)
- Hallucination prevention (says "I don't know" when answer isn't in docs)
- Conversation memory (remembers last 3 turns)
- 100% free APIs (Groq + Cohere + Gemini)

## Tech Stack

- **LLM:** Groq API (LLaMA 3.3 70B)
- **Embeddings:** Cohere embed-english-v3.0
- **Vector Database:** ChromaDB
- **Framework:** LangChain
- **UI:** Streamlit
- **Language:** Python

## Run Locally
```bash
git clone https://github.com/yourusername/company-knowledge-base
cd company-knowledge-base
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add your API keys to `.env`:
```
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
```
```bash
python ingest.py     # embed your documents
streamlit run app.py # launch the app
```

## Project Structure
```
├── ingest.py       # Document loading, chunking, embedding pipeline
├── retriever.py    # Vector similarity search
├── app.py          # Streamlit UI + RAG chain
├── data/           # Place your PDF documents here
└── chroma_db/      # Auto-generated vector store
```
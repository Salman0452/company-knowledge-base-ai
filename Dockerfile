# ── BASE IMAGE ─────────────────────────────────────────────────────────────────
# We use Python 3.11 slim — smaller than full Python image, faster to build
# "slim" means no unnecessary system packages included
FROM python:3.12-slim

# ── METADATA ───────────────────────────────────────────────────────────────────
# Good practice — documents who built this and what it is
LABEL maintainer="Salman Ahmad"
LABEL description="Company Knowledge Base RAG System"

# ── SYSTEM DEPENDENCIES ────────────────────────────────────────────────────────
# Some Python packages need these system libraries to compile
# We install them first before Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf cleans up apt cache — keeps image size small

# ── WORKING DIRECTORY ──────────────────────────────────────────────────────────
# All commands from here run inside /app inside the container
WORKDIR /app

# Increase pip network timeout to reduce ReadTimeout errors during install
ENV PIP_DEFAULT_TIMEOUT=120

# ── INSTALL PYTHON DEPENDENCIES ────────────────────────────────────────────────
# We copy requirements.txt FIRST (before copying app code)
# Why? Docker caches layers. If requirements don't change, this layer
# is reused on rebuild — makes rebuilds much faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir keeps image smaller (no pip cache stored)

# ── COPY APPLICATION CODE ──────────────────────────────────────────────────────
# Now copy the actual app — this layer changes often, so it comes last
COPY . .

# ── EXPOSE PORTS ───────────────────────────────────────────────────────────────
# Tell Docker which ports this container uses
# FastAPI = 8000, Streamlit = 8501
EXPOSE 8000 8501

# ── DEFAULT COMMAND ────────────────────────────────────────────────────────────
# This is overridden by docker-compose for each service
# But useful as a fallback default
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

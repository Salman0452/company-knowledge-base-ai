import uuid
import requests
import streamlit as st
import os
import time


# ── CONFIG ─────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ── PAGE SETUP ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Knowledge Base",
    page_icon="🏨",
)

st.title("Company Knowledge Base AI")

# ── SESSION STATE ──────────────────────────────────────────────────────────────
# session_id — unique per browser session, sent to FastAPI so it tracks memory
# messages   — full chat history for display
# doc_names  — list of docs currently in ChromaDB (fetched from backend)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

def fetch_documents():
    """Ask FastAPI for the current list of document names."""
    for attempt in range(3):  # try 3 times
        try:
            res = requests.get(f"{BACKEND_URL}/documents", timeout=15)
            if res.status_code == 200:
                return res.json()["documents"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < 2:
                time.sleep(3)  # wait 3 seconds before retrying
            else:
                st.warning("Backend not ready yet. Please refresh the page.")
    return []


# Load document list on startup (and after uploads)
if not st.session_state.doc_names:
    st.session_state.doc_names = fetch_documents()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.title("Document Manager")

# ── PDF UPLOAD ─────────────────────────────────────────────────────────────────
st.sidebar.subheader("Upload New PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload any PDF — it will be ingested and searchable immediately"
)

if uploaded_file is not None:
    if st.sidebar.button("Ingest PDF", type="primary"):
        with st.sidebar:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=60   # ingestion can take time for large PDFs
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"{data['message']}")
                        st.caption(f"{data['chunks_added']} chunks added")
                        # Refresh document list after successful upload
                        st.session_state.doc_names = data["total_docs"]
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend.")

st.sidebar.markdown("---")

# ── DOCUMENT FILTER ────────────────────────────────────────────────────────────
st.sidebar.subheader("Filter Documents")

doc_names = st.session_state.doc_names

if doc_names:
    selected_docs = st.sidebar.multiselect(
        "Search in:",
        options=doc_names,
        default=doc_names,
        help="Select which documents to search"
    )
    st.sidebar.caption(f"{len(doc_names)} document(s) loaded")
else:
    selected_docs = []
    st.sidebar.info("No documents yet. Upload a PDF above.")

st.sidebar.markdown("---")

# ── DELETE DOCUMENT ────────────────────────────────────────────────────────────
if doc_names:
    st.sidebar.subheader("Remove Document")
    doc_to_delete = st.sidebar.selectbox("Select document to remove:", options=[""] + doc_names)
    if doc_to_delete and st.sidebar.button("Delete", type="secondary"):
        with st.sidebar:
            with st.spinner("Deleting..."):
                try:
                    res = requests.delete(
                        f"{BACKEND_URL}/documents/{doc_to_delete}",
                        timeout=10
                    )
                    if res.status_code == 200:
                        st.success(f"Deleted '{doc_to_delete}'")
                        st.session_state.doc_names = fetch_documents()
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Delete failed"))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend.")

# ── CHAT HISTORY DISPLAY ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── HANDLE CHAT INPUT ──────────────────────────────────────────────────────────
if not doc_names:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    if prompt := st.chat_input("Ask about your documents..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/query",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                            "selected_docs": selected_docs
                        },
                        timeout=30
                    )

                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]

                        st.markdown(answer)

                        # Show sources — same as your original app
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(f"**{src['doc_name']}** — Page {src['page']}")
                                st.caption(src["preview"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })

                    else:
                        error_msg = res.json().get("detail", "Something went wrong.")
                        st.error(error_msg)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend. Is FastAPI running?")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The PDF might be large — try again.")






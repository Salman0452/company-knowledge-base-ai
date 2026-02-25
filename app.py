import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Knowledge Base",
    page_icon="🏨",
    layout="centered"
)

st.title("Company Knowledge Base AI")
st.caption("Ask anything about company HR policies and procedures.")

# ── LOAD VECTOR STORE & LLM (cached so it loads only once) ────────────────────
@st.cache_resource
def load_chain():
    # Load embeddings — must match what you used in ingest.py
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    # Load existing ChromaDB from disk
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Groq LLM — same model you used in Week 1
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,        # low temp = factual, consistent answers
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Memory — keeps last 3 conversation turns (you built this in Week 1)
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key = "chat_history",
        return_messages = True,
        output_key = "answer"
    )

    # ConversationalRetrievalChain ties everything together:
    # retriever finds relevant chunks → LLM answers based on chunks + history
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": 4}
        ),
        memory = memory,
        return_source_documents = True,      # we'll show sources to the user
        verbose = False
    )
    return chain

# ── SESSION STATE — chat history for display ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── DISPLAY EXISTING CHAT HISTORY ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── HANDLE NEW USER INPUT ──────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about HR policies..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge base..."):
            chain = load_chain()

            # This is the system prompt — controls how the LLM behaves
            # We inject it by modifying the question
            augmented_prompt = f"""You are an HR assistant. Answer based ONLY on 
the provided context. If the context does not contain the answer, say: 
'I could not find this information in the company documents.'
Do not make up information.

Question: {prompt}"""
            
            result = chain.invoke({"question": augmented_prompt})
            answer = result["answer"]
            source_docs = result["source_documents"]
        
        st.markdown(answer)

        # Show sources so users can verify — this builds trust
        with st.expander("Sources used"):
            for i, doc in enumerate(source_docs):
                page = doc.metadata.get("page", "?")
                source = doc.metadata.get("source", "unknown")
                st.markdown(f"**Chunk {i+1}** — {source}, Page {page}")
                st.caption(doc.page_content[:200] + "...")
        
    st.session_state.messages.append({"role": "assistant", "content": answer})






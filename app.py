import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()

# This works both locally (.env) and on Streamlit Cloud (st.secrets)
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Knowledge Base",
    page_icon="🏨",
)

st.title("Company Knowledge Base AI")

# ── LOAD RESOURCES ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=get_secret("COHERE_API_KEY")
    )
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,        # low temp = factual, consistent answers
            groq_api_key=get_secret("GROQ_API_KEY")
        )
vectorstore = load_vectorstore()
llm = load_llm()

# ── SIDEBAR — Document Filter ──────────────────────────────────────────────────
st.sidebar.title("Filter Documents")

# Get all unique document names from ChromaDB metadata
all_docs = vectorstore.get()["metadatas"]
doc_names = sorted(set(m["doc_name"] for m in all_docs if "doc_name" in m))

selected_docs = st.sidebar.multiselect(
    "Search in:",
    options=doc_names,
    default=doc_names,
    help="Select which documents to search"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(doc_names)} documents loaded")

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# ── DISPLAY CHAT HISTORY ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── HANDLE INPUT ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about company policies..."):

    st.session_state.messages.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            # Build retriever with metadata filter if specific docs selected
            if selected_docs and len(selected_docs) < len(doc_names):
                # Filter ChromaDB to only search selected documents
                retriever = vectorstore.as_retriever(
                    search_type = "similarity",
                    search_kwargs = {
                        "k": 4,
                        "filter": {"doc_name": {"$in": selected_docs}}
                    }
                )
            else:
                # No filter — search everything
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
            
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
                verbose=False
            )

            # This is the system prompt — controls how the LLM behaves
            # We inject it by modifying the question
            augmented_prompt = f"""You are a company policy assistant. 
Answer based ONLY on the provided context. 
If the answer is not in the context, say: 'I could not find this in the selected documents.'
Do not make up information.

Question: {prompt}"""
            
            result = chain.invoke({"question": augmented_prompt})
            answer = result["answer"]
            source_docs = result["source_documents"]
        
        st.markdown(answer)

        # Show sources so users can verify — this builds trust
        with st.expander("Sources"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**{doc.metadata.get('doc_name', 'Unknown')}** — Page {doc.metadata.get('page', '?')}")
                st.caption(doc.page_content[:200] + "...")
        
    st.session_state.messages.append({"role": "assistant", "content": answer})






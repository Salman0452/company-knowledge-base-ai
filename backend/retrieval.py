from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def load_retriever(k=4):
    """
    Loads the existing ChromaDB and returns a retriever.
    k = number of chunks to return per query
    """
    embedding_model = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    # as_retriever() wraps the vectorstore so LangChain chains can use it
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # cosine similarity search
        search_kwargs={"k": k}
    )

    return retriever


# ── QUICK TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = load_retriever()
    query = "What is the employee relocation policy?"

    results = retriever.invoke(query)

    print(f"\nTop {len(results)} chunks for: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"── Chunk {i+1} ─────────────────────────")
        print(doc.page_content[:300])
        print(f"Source: {doc.metadata.get('source', 'unknown')}, "
              f"Page: {doc.metadata.get('page', '?')}")
        print()
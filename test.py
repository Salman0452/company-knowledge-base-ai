print("Step 1: Python works")

from dotenv import load_dotenv
load_dotenv()
print("Step 2: dotenv works")

from langchain_community.document_loaders import PyPDFLoader
print("Step 3: document loaders work")

from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Step 4: text splitter works")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
print("Step 5: Google embeddings import works")

from langchain_community.vectorstores import Chroma
print("Step 6: Chroma works")

print("✅ All imports OK — crash is in execution, not imports")
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

chromadb.Settings(anonymized_telemetry=False)

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")
os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_chroma_collection(name: str = None):
    return Chroma(
        collection_name=name or COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

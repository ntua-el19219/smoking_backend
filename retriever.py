import os
from chromaClient import get_chroma_collection

os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE"

def retrieve_documents(user_id: str, query: str, k: int = 3):
    kb = get_chroma_collection("knowledge_base")
    history = get_chroma_collection("chat_history")

    kb_docs = kb.similarity_search(query, k=k)
    hist_docs = history.similarity_search(query, k=1, filter={"user_id": user_id})

    all_docs = kb_docs + hist_docs

    print("[DEBUG] Retrieved documents:")
    for doc in all_docs:
        print(f"- content: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return all_docs
    
def retrieve_chat_history(user_id: str, limit: int = 10) -> list[dict]:
    collection = get_chroma_collection("chat_history")
    all_items = collection.get(include=['metadatas', 'documents'])

    filtered = [
        {
            "timestamp": meta["timestamp"],
            "text": doc,
            "role": meta.get("role", "user")
        }
        for doc, meta in zip(all_items["documents"], all_items["metadatas"])
        if meta.get("user_id") == user_id
    ]

    sorted_items = sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
    return sorted_items[:limit][::-1]  # Από παλιότερα προς νεότερα

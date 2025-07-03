from chromaClient import get_chroma_collection
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from retriever import retrieve_chat_history

def append_to_history(user_id: str, role: str, text: str):
    collection = get_chroma_collection("chat_history")
    now = datetime.now(timezone.utc).isoformat()
    print(f"[DEBUG] Saving message: {text} (user_id={user_id}, role={role}, timestamp={now})")
    collection.add_texts(
        [text],
        metadatas=[{
            "user_id": user_id,
            "timestamp": now,
            "role": role
        }]
    )

def load_user_history(user_id: str, k: int = 3):
    collection = get_chroma_collection("chat_history")
    docs = collection.similarity_search_with_score("recent messages", k=k, filter={"user_id": user_id})

    history = []
    token_count = 0
    for doc, _ in docs:
        if token_count > 1000:
            break
        history.append(doc.page_content)
        token_count += len(doc.page_content.split())

    return "\n".join(history)

def load_user_history_as_list(user_id: str, k: int = 20):
    collection = get_chroma_collection("chat_history")
    results = collection.get(where={"user_id": user_id})
    
    items = []
    for content, meta in zip(results["documents"], results["metadatas"]):
        items.append({
            "content": content.strip(),
            "timestamp": meta.get("timestamp", ""),
            "role": meta.get("role", "user")  # default "user" για backward compat
        })

    items.sort(key=lambda x: x["timestamp"])
    return items[-k:]

def delete_user_history(user_id: str):
    collection = get_chroma_collection("chat_history")
    # Διαγράφει όλα τα documents με βάση το user_id
    deleted = collection.delete(where={"user_id": user_id})
    print(f"[DEBUG] Deleted history for user {user_id}")
    return deleted

def load_chat_history_for_prompt(user_id: str, limit: int = 10) -> List:
    messages = retrieve_chat_history(user_id, limit)
    parsed = []
    for msg in messages:
        if msg["role"] == "user":
            parsed.append(HumanMessage(content=msg["text"]))
        else:
            parsed.append(AIMessage(content=msg["text"]))
    return parsed

def get_formatted_memory(user_id: str, limit: int = 10) -> str:
    history = retrieve_chat_history(user_id, limit=limit)
    formatted = []
    for h in history:
        role = "User" if h["role"] == "user" else "Hailey"
        formatted.append(f"{role}: {h['text'].strip()}")
    return "\n".join(formatted)
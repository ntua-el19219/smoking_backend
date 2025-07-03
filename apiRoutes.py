from pydantic import BaseModel
from fastapi import APIRouter
from memory import append_to_history
from graph_builder import rag_agent_executor  # ✅ αλλαγή

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    input: str

@router.post("/chat/")
async def chat(request: ChatRequest):
    response = rag_agent_executor.invoke({
        "user_id": request.user_id,
        "input": request.input
    })

    final_response = response  # ✅ πλέον είναι σκέτο string

    print(f"[DEBUG] Cleaned response:\n{final_response}\n")

    append_to_history(request.user_id, "user", request.input)
    append_to_history(request.user_id, "ai", final_response)

    from memory import load_user_history
    full_history = load_user_history(request.user_id, k=10)

    return {
        "response": final_response,
        "history": full_history
    }

class SaveMessageRequest(BaseModel):
    user_id: str
    content: str
    role: str  # 'user' ή 'ai'

@router.post("/save/")
async def save_message(request: SaveMessageRequest):
    from memory import append_to_history
    append_to_history(request.user_id, request.role, request.content)
    return {"status": "saved"}

@router.get("/history/{user_id}")
async def get_history(user_id: str):
    from memory import load_user_history_as_list
    full_history = load_user_history_as_list(user_id, k=20)
    print(f"[DEBUG] Returned history: {full_history}")
    return {"history": full_history}

@router.delete("/history/{user_id}")
async def delete_history(user_id: str):
    from memory import delete_user_history
    delete_user_history(user_id)
    return {"status": "history deleted"}

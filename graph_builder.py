from langgraph.graph import StateGraph
from typing import TypedDict
import logging

from chains import (
    rag_chain,
    format_docs,
    classify_intent,
    detect_emotion,
    check_topic_relevance,
    generate_irrelevant_reply,
    generate_motivation,
    needs_clarification,
    fallback_chain,
    clarification_chain,
)

from memory import get_formatted_memory
from retriever import retrieve_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === STATE === #
class GraphState(TypedDict):
    user_id: str
    input: str
    intent: str
    emotion: str
    topic_relevance: str
    needs_clarification: bool
    memory: str
    response: str

# === NODES === #
def clarification_node(state: GraphState) -> GraphState:
    logger.info("Checking if clarification is needed...")
    if needs_clarification(state["input"]):
        reply = clarification_chain.invoke({"input": state["input"]})
        return {**state, "needs_clarification": True, "response": reply}
    return {**state, "needs_clarification": False}

def topic_check_node(state: GraphState) -> GraphState:
    logger.info("Checking topic relevance...")
    result = check_topic_relevance(state["input"])
    if result == "irrelevant":
        reply = generate_irrelevant_reply(state["input"])
        return {**state, "topic_relevance": "irrelevant", "response": reply}
    return {**state, "topic_relevance": "relevant"}

def intent_node(state: GraphState) -> GraphState:
    logger.info("Classifying intent...")
    intent = classify_intent(state["input"])
    return {**state, "intent": intent}

def memory_node(state: GraphState) -> GraphState:
    logger.info("Fetching memory...")
    memory = get_formatted_memory(state["user_id"], limit=10)
    return {**state, "memory": memory}

def emotion_node(state: GraphState) -> GraphState:
    logger.info("Detecting emotional state...")
    emotion = detect_emotion(state["input"])
    return {**state, "emotion": emotion}

def rag_response_node(state: GraphState) -> GraphState:
    logger.info("Generating RAG-based answer...")
    docs = retrieve_documents(state["user_id"], state["input"])
    context = format_docs(docs, max_tokens=1000)
    combined_context = state.get("memory", "") + "\n\n" + context

    reply = rag_chain.invoke({
        "context": combined_context,
        "question": state["input"]
    })
    return {**state, "response": reply}

def fallback_response_node(state: GraphState) -> GraphState:
    logger.info("Generating fallback emotional support answer...")
    reply = fallback_chain.invoke({"question": state["input"]})
    return {**state, "response": reply}

def motivation_node(state: GraphState) -> GraphState:
    logger.info("Checking for motivation need...")
    if state["emotion"] in [
        "sadness", "fear", "anxiety", "shame", "guilt", "frustration", "loneliness"
    ]:
        motivation = generate_motivation(state["input"])
        return {**state, "response": state["response"] + "\n\n" + motivation}
    return state

def output_node(state: GraphState) -> str:
    logger.info("Final response ready.")
    return state["response"]

# === GRAPH === #
graph = StateGraph(GraphState)
graph.set_entry_point("clarification")

graph.add_node("clarification", clarification_node)
graph.add_node("topic_check", topic_check_node)
graph.add_node("intent", intent_node)
graph.add_node("memory", memory_node)
graph.add_node("emotion", emotion_node)
graph.add_node("rag_response", rag_response_node)
graph.add_node("fallback_response", fallback_response_node)
graph.add_node("motivation", motivation_node)
graph.add_node("output", output_node)

# === EDGES === #
graph.add_conditional_edges("clarification", lambda s: "output" if s["needs_clarification"] else "topic_check")
graph.add_conditional_edges("topic_check", lambda s: "output" if s["topic_relevance"] == "irrelevant" else "intent")
graph.add_edge("intent", "memory")
graph.add_edge("memory", "emotion")
graph.add_conditional_edges("emotion", lambda s: "rag_response" if s["intent"] == "use_rag" else "fallback_response")
graph.add_edge("rag_response", "motivation")
graph.add_edge("fallback_response", "motivation")
graph.add_edge("motivation", "output")
graph.set_finish_point("output")

# === COMPILE === #
rag_agent_executor = graph.compile()

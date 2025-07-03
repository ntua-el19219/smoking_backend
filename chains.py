from langchain_core.output_parsers import StrOutputParser
from config import llm
from prompt_templates import (
    RAG_PROMPT,
    FALLBACK_PROMPT,
    CLARIFICATION_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    EMOTION_DETECTION_PROMPT,
    TOPIC_RELEVANCE_PROMPT,
    IRRELEVANT_TOPIC_PROMPT,
    MOTIVATION_PROMPT
)

# === Shared parser ===
parser = StrOutputParser()

# === RAG ===
rag_chain = RAG_PROMPT | llm
def format_docs(docs, max_tokens=1000):
    result = []
    token_count = 0
    for doc in docs:
        content = doc.page_content.strip()
        if not content or len(content.split()) < 5:
            continue
        if content.startswith("%PDF") or "obj" in content[:100]:
            continue
        if "User:" in content or "AI:" in content:
            continue

        words = content.split()
        if token_count + len(words) > max_tokens:
            break

        result.append(content)
        token_count += len(words)

    return "\n\n---\n\n".join(result)

# === Clarification ===
clarification_chain = CLARIFICATION_PROMPT | llm | parser
def needs_clarification(input: str) -> bool:
    response = clarification_chain.invoke({"input": input}).strip()
    return any(x in response.lower() for x in [
        "could you clarify", "can you tell me", "help me understand",
        "could you explain", "not sure what you mean", "can you clarify"
    ])

# === Intent Classification ===
classify_intent_chain = INTENT_CLASSIFICATION_PROMPT | llm | parser
def classify_intent(input: str) -> str:
    return classify_intent_chain.invoke({"input": input}).strip()

# === Emotion Detection ===
detect_emotion_chain = EMOTION_DETECTION_PROMPT | llm | parser
def detect_emotion(input: str) -> str:
    return detect_emotion_chain.invoke({"input": input}).strip()

# === Topic Relevance ===
check_topic_relevance_chain = TOPIC_RELEVANCE_PROMPT | llm | parser
def check_topic_relevance(input: str) -> str:
    return check_topic_relevance_chain.invoke({"input": input}).strip()

# === Irrelevant topic fallback ===
irrelevant_topic_chain = IRRELEVANT_TOPIC_PROMPT | llm | parser
def generate_irrelevant_reply(input: str) -> str:
    return irrelevant_topic_chain.invoke({"input": input}).strip()

# === Motivation ===
generate_motivation_chain = MOTIVATION_PROMPT | llm | parser
def generate_motivation(context: str) -> str:
    return generate_motivation_chain.invoke({"context": context}).strip()

# === Fallback response ===
fallback_chain = FALLBACK_PROMPT | llm | parser

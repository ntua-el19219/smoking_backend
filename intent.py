from langchain_core.output_parsers import StrOutputParser

from prompt_templates import (
    INTENT_CLASSIFICATION_PROMPT,
    EMOTION_DETECTION_PROMPT,
    TOPIC_RELEVANCE_PROMPT,
    IRRELEVANT_TOPIC_PROMPT,
    MOTIVATION_PROMPT,
    CLARIFICATION_PROMPT
)

from config import llm

parser = StrOutputParser()

def classify_intent(input: str) -> str:
    chain = INTENT_CLASSIFICATION_PROMPT | llm | parser
    return chain.invoke({"input": input}).strip()

def detect_emotion(input: str) -> str:
    chain = EMOTION_DETECTION_PROMPT | llm | parser
    return chain.invoke({"input": input}).strip()

def check_topic_relevance(input: str) -> str:
    chain = TOPIC_RELEVANCE_PROMPT | llm | parser
    return chain.invoke({"input": input}).strip()

def generate_irrelevant_reply(input: str) -> str:
    chain = IRRELEVANT_TOPIC_PROMPT | llm | parser
    return chain.invoke({"input": input}).strip()

def generate_motivation(context: str) -> str:
    chain = MOTIVATION_PROMPT | llm | parser
    return chain.invoke({"context": context}).strip()

def needs_clarification(input: str) -> bool:
    chain = CLARIFICATION_PROMPT | llm | parser
    response = chain.invoke({"input": input}).strip()

    # Απλή heuristic: Αν η απάντηση περιέχει φράσεις ερώτησης, θεωρούμε πως χρειάζεται διευκρίνιση
    if any(x in response.lower() for x in ["could you clarify", "can you tell me", "help me understand", "could you explain", "not sure what you mean", "can you clarify"]):
        return True
    return False

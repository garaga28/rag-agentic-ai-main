"""LangGraph RAG workflow: retrieve -> generate -> grade, with a retry loop and a refusal path.

                    +-----------------------------+
                    |                             |  not grounded, attempts left
                    v                             |
 START -> retrieve -+-> generate -> grade --------+
                    |       |          |
    nothing relevant|       |unanswer- | grounded
                    v       v  able    v
                  refuse <--+       finalize -> END
                    |
                    +--> END

Greetings ("hi", "thanks") are answered by a greet node without retrieval.
confidence_score = 0.7 * groundedness (LLM grader) + 0.3 * best retrieval similarity.
A refusal always has confidence_score 0.0.
"""

import re
import time
from typing import Callable, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config


GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hii+|good (morning|afternoon|evening)|thanks?( you)?|thank you|"
    r"ok(ay)?|bye|how are you)\W*$",
    re.IGNORECASE,
)


class RetrievedChunk(TypedDict):
    text: str
    page_number: int
    score: float  # cosine similarity, higher is more similar


# A retriever takes the question and returns the top-k chunks.
Retriever = Callable[[str], list[RetrievedChunk]]


class AgentState(TypedDict, total=False):
    question: str
    chunks: list[RetrievedChunk]  # relevant chunks only
    context: list[str]  # chunk texts, returned to the caller
    answer: str
    answerable: bool
    groundedness: float
    grounded: bool
    feedback: str
    attempts: int
    score: float


class GeneratedAnswer(BaseModel):
    answerable: bool = Field(
        description="True only if the context contains the information needed to answer."
    )
    answer: str = Field(description="The answer, written only from the context.")


class GroundingGrade(BaseModel):
    groundedness: float = Field(
        description="Number from 0.0 to 1.0: fraction of the answer's claims that are directly supported by the context.",
    )
    grounded: bool = Field(description="True if every material claim is supported by the context.")
    unsupported_claims: str = Field(description="Claims not supported by the context, or 'none'.")


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about the eBook 'Agentic AI: An Executive's Guide'. "
     "Use ONLY the numbered context passages below. Do not use outside knowledge, "
     "even if you know the answer.\n"
     "Set answerable to false (and leave the answer empty) only when the passages contain "
     "nothing relevant to the question. If they contain relevant information, answer with "
     "what they say, even if it is not phrased exactly as the question asks (for example, "
     "there may be no single sentence labelled as a definition).\n"
     "When answerable, write a clear, complete answer in plain text or Markdown "
     "(bullets are fine for lists, never HTML) "
     "and cite the page each fact comes from, like (p. 12), using the page number "
     "shown in the passage header (not the passage number).\n\n"
     "Context:\n{context}"),
    ("human", "{question}{feedback}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Judge whether the ANSWER is supported by the CONTEXT. "
     "Ignore whether the answer is true in the real world; only the context counts. "
     "Page citations are not claims.\n\nCONTEXT:\n{context}"),
    ("human", "QUESTION: {question}\n\nANSWER: {answer}"),
])


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"--- Passage {i} (from page {c['page_number']}) ---\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def pinecone_retriever(k: int = config.TOP_K) -> Retriever:
    """Top-k cosine search against the Pinecone index built by src/ingestion.py."""
    from langchain_pinecone import PineconeVectorStore

    from src.models import get_embeddings

    config.require_keys("PINECONE_API_KEY")
    store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME, embedding=get_embeddings()
    )

    def retrieve(question: str) -> list[RetrievedChunk]:
        for attempt in range(3):  # ride out dropped connections to the embedding API
            try:
                results = store.similarity_search_with_score(question, k=k)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        return [
            {
                "text": doc.page_content,
                "page_number": int(doc.metadata.get("page_number", 0)),
                "score": float(score),
            }
            for doc, score in results
        ]

    return retrieve


def default_llm():
    from src.models import get_chat_models

    return get_chat_models()


def structured(llm, schema):
    """Structured-output runnable; a list of models becomes a primary with fallbacks."""
    models = llm if isinstance(llm, list) else [llm]
    runnables = [m.with_structured_output(schema) for m in models]
    return runnables[0].with_fallbacks(runnables[1:]) if len(runnables) > 1 else runnables[0]


def build_rag_graph(retriever: Retriever | None = None, llm=None):
    """Compile the graph. retriever and llm are injectable so the graph can be tested offline."""
    retriever = retriever or pinecone_retriever()
    llm = llm or default_llm()
    generator = GENERATE_PROMPT | structured(llm, GeneratedAnswer)
    grader = GRADE_PROMPT | structured(llm, GroundingGrade)

    def retrieve(state: AgentState) -> AgentState:
        results = retriever(state["question"])
        relevant = [c for c in results if c["score"] >= config.MIN_RELEVANCE]
        # The caller gets exactly the chunks the answer is based on.
        return {"chunks": relevant, "context": [c["text"] for c in relevant], "attempts": 0}

    def generate(state: AgentState) -> AgentState:
        feedback = state.get("feedback", "")
        out: GeneratedAnswer = generator.invoke({
            "context": format_context(state["chunks"]),
            "question": state["question"],
            "feedback": f"\n\n(Your previous answer was rejected: {feedback} "
                        "Answer again using only the context.)" if feedback else "",
        })
        return {
            "answer": out.answer.strip(),
            "answerable": out.answerable and bool(out.answer.strip()),
            "attempts": state.get("attempts", 0) + 1,
        }

    def grade(state: AgentState) -> AgentState:
        g: GroundingGrade = grader.invoke({
            "context": format_context(state["chunks"]),
            "question": state["question"],
            "answer": state["answer"],
        })
        groundedness = min(max(g.groundedness, 0.0), 1.0)
        return {
            "groundedness": groundedness,
            "grounded": g.grounded and groundedness >= 0.7,
            "feedback": f"unsupported claims: {g.unsupported_claims}",
        }

    def finalize(state: AgentState) -> AgentState:
        best_similarity = max((c["score"] for c in state["chunks"]), default=0.0)
        best_similarity = min(max(best_similarity, 0.0), 1.0)
        score = 0.7 * state["groundedness"] + 0.3 * best_similarity
        return {"score": round(score, 2)}

    def refuse(state: AgentState) -> AgentState:
        # A refusal is not based on any passage, so no chunks are returned.
        return {"answer": config.REFUSAL_MESSAGE, "context": [], "chunks": [], "score": 0.0}

    def greet(state: AgentState) -> AgentState:
        return {"answer": config.GREETING_MESSAGE, "context": [], "chunks": [], "score": 0.0}

    def route_start(state: AgentState) -> str:
        return "greet" if GREETING_RE.match(state["question"]) else "retrieve"

    def after_retrieve(state: AgentState) -> str:
        return "generate" if state["chunks"] else "refuse"

    def after_generate(state: AgentState) -> str:
        return "grade" if state["answerable"] else "refuse"

    def after_grade(state: AgentState) -> str:
        if state["grounded"]:
            return "finalize"
        if state["attempts"] < config.MAX_GENERATION_ATTEMPTS:
            return "generate"
        return "refuse"

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("grade", grade)
    workflow.add_node("finalize", finalize)
    workflow.add_node("refuse", refuse)
    workflow.add_node("greet", greet)

    workflow.add_conditional_edges(START, route_start, ["greet", "retrieve"])
    workflow.add_conditional_edges("retrieve", after_retrieve, ["generate", "refuse"])
    workflow.add_conditional_edges("generate", after_generate, ["grade", "refuse"])
    workflow.add_conditional_edges("grade", after_grade, ["finalize", "generate", "refuse"])
    workflow.add_edge("finalize", END)
    workflow.add_edge("refuse", END)
    workflow.add_edge("greet", END)

    return workflow.compile()


def run(graph, query: str) -> tuple[dict, list[RetrievedChunk]]:
    """Run one query; return the required JSON payload plus the chunks with page numbers."""
    result = graph.invoke({"question": query})
    payload = {
        "query": query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result.get("context", []),
        "confidence_score": float(result.get("score", 0.0)),
    }
    return payload, result.get("chunks", [])


def ask(graph, query: str) -> dict:
    """Run one query and return the required JSON payload."""
    return run(graph, query)[0]

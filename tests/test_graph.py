"""Offline tests for the graph routing and the API payload (no API keys needed).

Run:  pytest -q
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

import app as app_module
from src import config
from src.graph import GeneratedAnswer, GroundingGrade, ask, build_rag_graph
from src.ingestion import chunk_documents, load_pages


@pytest.fixture(autouse=True)
def fixed_relevance_cutoff(monkeypatch):
    # The default depends on the configured provider; pin it so the tests do not.
    monkeypatch.setattr(config, "MIN_RELEVANCE", 0.25)


class FakeStructured:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return self.outputs.pop(0)


class FakeLLM:
    """Returns scripted structured outputs for the generator and the grader."""

    def __init__(self, answers=(), grades=()):
        self.gen = FakeStructured(answers)
        self.grade = FakeStructured(grades)

    def with_structured_output(self, schema):
        fake = self.gen if schema is GeneratedAnswer else self.grade
        return RunnableLambda(fake.invoke)


def retriever_with(score):
    return lambda q: [
        {"text": "Agentic AI systems plan and act autonomously.", "page_number": 4, "score": score},
        {"text": "Agents use memory, tools and planning.", "page_number": 9, "score": score - 0.05},
    ]


def test_grounded_answer_returns_payload_and_score():
    llm = FakeLLM(
        answers=[GeneratedAnswer(answerable=True, answer="Agentic AI plans and acts (p. 4).")],
        grades=[GroundingGrade(groundedness=1.0, grounded=True, unsupported_claims="none")],
    )
    out = ask(build_rag_graph(retriever_with(0.6), llm), "What is Agentic AI?")
    assert set(out) == {"query", "final_answer", "retrieved_context_chunks", "confidence_score"}
    assert out["final_answer"].startswith("Agentic AI plans")
    assert len(out["retrieved_context_chunks"]) == 2
    assert out["confidence_score"] == round(0.7 * 1.0 + 0.3 * 0.6, 2)


def test_irrelevant_retrieval_refuses_without_calling_llm():
    llm = FakeLLM()
    out = ask(build_rag_graph(retriever_with(0.1), llm), "What is the capital of France?")
    assert out["final_answer"] == config.REFUSAL_MESSAGE
    assert out["confidence_score"] == 0.0
    assert llm.gen.calls == 0


def test_greeting_gets_friendly_reply_without_retrieval():
    llm = FakeLLM()

    def no_retrieval(q):
        raise AssertionError("greetings must not hit the retriever")

    out = ask(build_rag_graph(no_retrieval, llm), "hi")
    assert out["final_answer"] == config.GREETING_MESSAGE
    assert out["retrieved_context_chunks"] == []
    assert out["confidence_score"] == 0.0


def test_only_relevant_chunks_are_returned():
    def mixed(q):
        return [
            {"text": "relevant", "page_number": 4, "score": 0.8},
            {"text": "noise", "page_number": 1, "score": 0.1},
        ]

    llm = FakeLLM(
        answers=[GeneratedAnswer(answerable=True, answer="An answer (p. 4).")],
        grades=[GroundingGrade(groundedness=1.0, grounded=True, unsupported_claims="none")],
    )
    out = ask(build_rag_graph(mixed, llm), "What is Agentic AI?")
    assert out["retrieved_context_chunks"] == ["relevant"]


def test_llm_says_unanswerable_refuses():
    llm = FakeLLM(answers=[GeneratedAnswer(answerable=False, answer="")])
    out = ask(build_rag_graph(retriever_with(0.5), llm), "Who won the 2022 FIFA World Cup?")
    assert out["final_answer"] == config.REFUSAL_MESSAGE
    assert out["retrieved_context_chunks"] == []
    assert out["confidence_score"] == 0.0
    assert llm.grade.calls == 0


def test_ungrounded_answer_is_retried_then_accepted():
    llm = FakeLLM(
        answers=[
            GeneratedAnswer(answerable=True, answer="Agentic AI was invented in 1950."),
            GeneratedAnswer(answerable=True, answer="Agentic AI plans and acts (p. 4)."),
        ],
        grades=[
            GroundingGrade(groundedness=0.2, grounded=False, unsupported_claims="invented in 1950"),
            GroundingGrade(groundedness=0.9, grounded=True, unsupported_claims="none"),
        ],
    )
    out = ask(build_rag_graph(retriever_with(0.5), llm), "What is Agentic AI?")
    assert llm.gen.calls == 2
    assert out["final_answer"] == "Agentic AI plans and acts (p. 4)."
    assert out["confidence_score"] > 0


def test_persistently_ungrounded_answer_is_refused():
    bad = GroundingGrade(groundedness=0.1, grounded=False, unsupported_claims="everything")
    llm = FakeLLM(
        answers=[GeneratedAnswer(answerable=True, answer="made up")] * config.MAX_GENERATION_ATTEMPTS,
        grades=[bad] * config.MAX_GENERATION_ATTEMPTS,
    )
    out = ask(build_rag_graph(retriever_with(0.5), llm), "What is Agentic AI?")
    assert out["final_answer"] == config.REFUSAL_MESSAGE
    assert llm.gen.calls == config.MAX_GENERATION_ATTEMPTS


def test_api_chat_endpoint(monkeypatch):
    llm = FakeLLM(
        answers=[GeneratedAnswer(answerable=True, answer="Agentic AI plans and acts (p. 4).")],
        grades=[GroundingGrade(groundedness=1.0, grounded=True, unsupported_claims="none")],
    )
    monkeypatch.setattr(app_module, "graph", build_rag_graph(retriever_with(0.6), llm))
    with TestClient(app_module.app) as client:
        resp = client.post("/chat", json={"query": "What is Agentic AI?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "What is Agentic AI?"
        assert body["confidence_score"] == 0.88
        assert client.post("/chat", json={"query": ""}).status_code == 422


def test_pdf_chunking_keeps_page_metadata():
    chunks = chunk_documents(load_pages())
    assert len(chunks) > 50
    assert all(config.MIN_CHUNK_CHARS <= len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    assert not any("Scan this QR code" in c.page_content for c in chunks)
    assert all(c.metadata["page_number"] >= 1 for c in chunks)

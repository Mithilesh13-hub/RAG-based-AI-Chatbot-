from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.config import REFUSAL_MESSAGE
from src.graph import (GenerationError, RetrievalError, build_graph, overall_relevance, run_graph,
                       term_coverage)
from tests.test_ingestion import make_settings


class FakeStore:
    def __init__(self, results=None, error=None):
        self.results, self.error = results or [], error

    def similarity_search_with_score(self, q, k=5):
        if self.error:
            raise self.error
        return self.results[:k]


class FakeLLM:
    def __init__(self, reply="Agentic AI is systems that act autonomously (page 3).", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, messages):
        if self.error:
            raise self.error
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)


def doc(text, page, score, cid="c"):
    return (Document(page_content=text, metadata={"source": "Ebook-Agentic-AI.pdf", "page": float(page),
                                                  "chunk_id": cid}), score)


GOOD = [doc("Agentic AI refers to autonomous AI agents that plan and act.", 3, 0.62, "a"),
        doc("Agentic AI systems use memory and tools.", 4, 0.55, "b"),
        doc("Unrelated appendix.", 9, 0.12, "z")]


def graph(store, llm, **kw):
    return build_graph(vectorstore=store, llm=llm, settings=make_settings(**kw))


def test_sufficient_path():
    llm = FakeLLM()
    out = run_graph("What is Agentic AI?", graph(FakeStore(GOOD), llm))
    assert out["final_answer"].startswith("Agentic AI is")
    assert out["confidence_score"] == pytest.approx(0.585)  # mean(0.62, 0.55)
    assert [c["page"] for c in out["citations"]] == [3, 4]  # low-score chunk not cited
    assert len(out["retrieved_context"]) == 3 and out["retrieved_context"][0]["page"] == 3
    # retrieved text is fenced off from instructions
    assert "<document_context>" in llm.calls[0][1].content
    assert "autonomous AI agents" not in llm.calls[0][0].content  # chunk text never enters system prompt
    assert "autonomous AI agents" in llm.calls[0][1].content


def test_low_scores_refuse_without_calling_llm():
    llm = FakeLLM()
    out = run_graph("Who won the 2022 FIFA World Cup?", graph(FakeStore([doc("misc", 1, 0.15)]), llm))
    assert out["final_answer"] == REFUSAL_MESSAGE
    assert out["confidence_score"] == 0.0 and out["citations"] == [] and llm.calls == []


def test_high_score_but_no_term_overlap_refuses():
    store = FakeStore([doc("Cooking recipes and kitchen tools.", 2, 0.5)])
    out = run_graph("Who won the 2022 FIFA World Cup?", graph(store, FakeLLM()))
    assert out["final_answer"] == REFUSAL_MESSAGE


def test_no_results_refuses():
    assert run_graph("What is Agentic AI?", graph(FakeStore([]), FakeLLM()))["final_answer"] == REFUSAL_MESSAGE


def test_model_refusal_zeroes_score_and_citations():
    out = run_graph("What is Agentic AI?", graph(FakeStore(GOOD), FakeLLM(reply=REFUSAL_MESSAGE)))
    assert out["final_answer"] == REFUSAL_MESSAGE
    assert out["confidence_score"] == 0.0 and out["citations"] == []


def test_configurable_threshold():
    out = run_graph("What is Agentic AI?", graph(FakeStore(GOOD), FakeLLM(), relevance_threshold=0.9))
    assert out["final_answer"] == REFUSAL_MESSAGE


def test_empty_question():
    with pytest.raises(ValueError):
        run_graph("   ", graph(FakeStore(GOOD), FakeLLM()))


def test_retrieval_failure_is_wrapped():
    with pytest.raises(RetrievalError):
        run_graph("q?", graph(FakeStore(error=RuntimeError("index missing")), FakeLLM()))


def test_generation_failure_is_wrapped():
    with pytest.raises(GenerationError):
        run_graph("What is Agentic AI?", graph(FakeStore(GOOD), FakeLLM(error=RuntimeError("down"))))


def test_scoring_helpers():
    assert overall_relevance([]) == 0.0
    assert term_coverage("What is Agentic AI?", "agentic ai systems") == 1.0
    assert term_coverage("Who won the 2022 FIFA World Cup?", "agentic ai systems") == 0.0
    assert term_coverage("the of a", "anything") == 0.0

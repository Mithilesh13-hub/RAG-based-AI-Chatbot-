from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from src import ingestion
from src.config import EMBEDDING_DIMENSION, ConfigurationError, Settings
from src.ingestion import IngestionError


def make_settings(**kw):
    base = dict(openai_api_key="k", pinecone_api_key="k", pinecone_index_name="idx", pinecone_cloud="aws",
                pinecone_region="us-east-1", embedding_model="m", chat_model="c", top_k=5,
                relevance_threshold=0.3, min_term_coverage=0.3, api_base_url="http://x")
    base.update(kw)
    return Settings(**base)


def blank_pdf(path):
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        w.write(fh)
    return path


# ---- validation
def test_missing_pdf(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        ingestion.validate_pdf(tmp_path / "nope.pdf")


def test_not_a_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_text("hello")
    with pytest.raises(IngestionError, match="not a valid PDF"):
        ingestion.validate_pdf(p)


def test_empty_file(tmp_path):
    p = tmp_path / "e.pdf"
    p.write_bytes(b"")
    with pytest.raises(IngestionError, match="empty"):
        ingestion.validate_pdf(p)


def test_corrupt_pdf(tmp_path):
    p = tmp_path / "c.pdf"
    p.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(IngestionError):
        ingestion.validate_pdf(p)


def test_valid_pdf(tmp_path):
    assert ingestion.validate_pdf(blank_pdf(tmp_path / "ok.pdf")).name == "ok.pdf"


# ---- extraction
def test_load_pages_skips_empty_and_numbers_from_one(tmp_path, monkeypatch):
    pdf = blank_pdf(tmp_path / "book.pdf")
    raw = [Document(page_content="  ", metadata={"page": 0}),
           Document(page_content="Agentic AI text", metadata={"page": 1})]
    monkeypatch.setattr(ingestion, "PyPDFLoader", lambda _p: SimpleNamespace(load=lambda: raw))
    pages = ingestion.load_pages(pdf)
    assert len(pages) == 1
    assert pages[0].metadata == {"source": "book.pdf", "page": 2}


def test_load_pages_all_empty(tmp_path, monkeypatch):
    pdf = blank_pdf(tmp_path / "book.pdf")
    monkeypatch.setattr(ingestion, "PyPDFLoader",
                        lambda _p: SimpleNamespace(load=lambda: [Document(page_content="", metadata={"page": 0})]))
    with pytest.raises(IngestionError, match="No extractable text"):
        ingestion.load_pages(pdf)


def test_load_pages_extraction_failure(tmp_path, monkeypatch):
    pdf = blank_pdf(tmp_path / "book.pdf")

    def boom():
        raise RuntimeError("bad")
    monkeypatch.setattr(ingestion, "PyPDFLoader", lambda _p: SimpleNamespace(load=boom))
    with pytest.raises(IngestionError, match="extraction failed"):
        ingestion.load_pages(pdf)


# ---- chunking
def test_chunking_and_deterministic_ids():
    text = " ".join(f"word{i}" for i in range(600))
    pages = [Document(page_content=text, metadata={"source": "b.pdf", "page": 1})]
    a, b = ingestion.split_documents(pages), ingestion.split_documents(pages)
    assert len(a) > 1
    assert all(len(c.page_content) <= ingestion.CHUNK_SIZE for c in a)
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]
    assert len({c.metadata["chunk_id"] for c in a}) == len(a)
    assert a[0].metadata["source"] == "b.pdf" and a[0].metadata["page"] == 1


def test_chunk_id_changes_with_content():
    assert ingestion.make_chunk_id("a", 1, 0, "x") != ingestion.make_chunk_id("a", 1, 0, "y")


# ---- Pinecone
class FakeIndex:
    def __init__(self):
        self.store = {}

    def fetch(self, ids):
        return SimpleNamespace(vectors={i: 1 for i in ids if i in self.store})

    def upsert(self, vectors):
        for v in vectors:
            self.store[v["id"]] = v


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]


def test_upsert_is_idempotent():
    chunks = ingestion.split_documents(
        [Document(page_content="alpha beta " * 400, metadata={"source": "b.pdf", "page": 1})])
    index, emb = FakeIndex(), FakeEmbeddings()
    first = ingestion.upsert_chunks(index, emb, chunks, batch_size=3, sleep=lambda _s: None)
    second = ingestion.upsert_chunks(index, emb, chunks, batch_size=3, sleep=lambda _s: None)
    assert first["added"] == len(chunks) and first["skipped"] == 0
    assert second["added"] == 0 and second["skipped"] == len(chunks)
    assert len(index.store) == len(chunks)
    meta = next(iter(index.store.values()))["metadata"]
    assert set(meta) == {"text", "source", "page", "chunk_id"}


def test_retry_recovers_then_fails():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("transient")
        return "ok"
    assert ingestion.with_retries(flaky, sleep=lambda _s: None) == "ok"
    with pytest.raises(RuntimeError):
        ingestion.with_retries(lambda: (_ for _ in ()).throw(RuntimeError("x")), retries=2, sleep=lambda _s: None)


class FakePinecone:
    def __init__(self, exists=False, dim=EMBEDDING_DIMENSION, ready_after=2):
        self.exists, self.dim, self.polls, self.ready_after = exists, dim, 0, ready_after
        self.created = None

    def has_index(self, name):
        return self.exists

    def create_index(self, **kw):
        self.created = kw

    def describe_index(self, name):
        self.polls += 1
        return SimpleNamespace(dimension=self.dim, status={"ready": self.polls >= self.ready_after})


def test_ensure_index_creates_and_waits():
    pc = FakePinecone()
    ingestion.ensure_index(pc, make_settings(), sleep=lambda _s: None)
    assert pc.created["dimension"] == 1536 and pc.created["metric"] == "cosine"
    assert pc.polls >= 2


def test_ensure_index_dimension_mismatch():
    with pytest.raises(IngestionError, match="dimension"):
        ingestion.ensure_index(FakePinecone(exists=True, dim=768), make_settings(), sleep=lambda _s: None)


def test_ingest_requires_credentials(tmp_path):
    with pytest.raises(ConfigurationError):
        ingestion.ingest(tmp_path / "x.pdf", settings=make_settings(openai_api_key=""))


def test_ingest_missing_pdf(tmp_path):
    with pytest.raises(IngestionError):
        ingestion.ingest(tmp_path / "x.pdf", settings=make_settings())


def test_download_rejects_bad_url(tmp_path):
    with pytest.raises(IngestionError):
        ingestion.download_pdf("https://example.com/nothing", tmp_path / "a.pdf")

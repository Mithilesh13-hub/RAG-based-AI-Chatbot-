"""PDF -> chunks -> embeddings -> Pinecone. Run with:  python -m src.ingestion"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from src.config import (
    DEFAULT_PDF_PATH,
    EMBEDDING_DIMENSION,
    PDF_DRIVE_URL,
    ConfigurationError,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
UPSERT_BATCH_SIZE = 100
MAX_RETRIES = 4

T = TypeVar("T")


class IngestionError(RuntimeError):
    """Raised for unrecoverable ingestion problems (bad PDF, index problems...)."""


# --------------------------------------------------------------------------- utils
def with_retries(fn: Callable[[], T], *, retries: int = MAX_RETRIES, base_delay: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep) -> T:
    """Run ``fn`` with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised after the last attempt
            if attempt == retries:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("Attempt %d/%d failed (%s); retrying in %.1fs", attempt, retries, exc, delay)
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ------------------------------------------------------------------------- download
def _drive_file_id(url: str) -> Optional[str]:
    match = re.search(r"/d/([A-Za-z0-9_-]+)", url) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    return match.group(1) if match else None


def download_pdf(url: str = PDF_DRIVE_URL, dest: Path = DEFAULT_PDF_PATH, timeout: int = 60) -> Path:
    """Download a public Google Drive PDF. Raises IngestionError if it is not accessible."""
    file_id = _drive_file_id(url)
    if not file_id:
        raise IngestionError(f"Could not extract a Google Drive file id from {url!r}.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    endpoint = "https://drive.google.com/uc"
    try:
        resp = session.get(endpoint, params={"export": "download", "id": file_id}, timeout=timeout)
        if "text/html" in resp.headers.get("Content-Type", ""):
            # Large files show a virus-scan interstitial: retry with the confirm token.
            match = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
            token = match.group(1) if match else "t"
            resp = session.get(endpoint, params={"export": "download", "id": file_id, "confirm": token},
                               timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise IngestionError(
            f"Could not download the PDF ({exc.__class__.__name__}). Download it manually and save it as {dest}."
        ) from exc
    if not resp.content.startswith(b"%PDF"):
        raise IngestionError(
            f"The download did not return a PDF (file may be private). Save it manually as {dest}."
        )
    dest.write_bytes(resp.content)
    return dest


# ------------------------------------------------------------------------ PDF -> docs
def validate_pdf(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise IngestionError(f"PDF not found at {path}. Place the eBook there or use --url.")
    if path.stat().st_size == 0:
        raise IngestionError(f"PDF at {path} is empty.")
    with path.open("rb") as fh:
        if not fh.read(5).startswith(b"%PDF"):
            raise IngestionError(f"{path} is not a valid PDF file.")
    try:
        page_count = len(PdfReader(str(path)).pages)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"PDF at {path} could not be read: {exc.__class__.__name__}.") from exc
    if page_count == 0:
        raise IngestionError(f"PDF at {path} has no pages.")
    return path


def load_pages(path: Path) -> List[Document]:
    """Extract text page by page, using OCR for scanned PDFs."""
    path = validate_pdf(path)
    try:
        raw_pages = PyPDFLoader(str(path)).load()
        pages = [
            Document(page_content=text, metadata={"source": path.name, "page": int(doc.metadata.get("page", 0)) + 1})
            for doc in raw_pages
            if (text := (doc.page_content or "").strip())
        ]
        if pages:
            return pages
    except Exception:
        pass
    
    # Fall back to OCR with easyocr + PyMuPDF
    logger.info("PyPDF extraction failed or empty; trying OCR with easyocr...")
    try:
        import easyocr
        from PIL import Image
        import pymupdf
        pdf = pymupdf.open(str(path))
        reader = easyocr.Reader(['en'], gpu=False)
        pages = []
        for i, page in enumerate(pdf, start=1):
            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                results = reader.readtext(img)
                text = "\n".join([line[1] for line in results if line[1].strip()]).strip()
                if text:
                    pages.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
                else:
                    logger.info("Page %d is blank after OCR", i)
            except Exception as e:
                logger.warning("Failed to OCR page %d: %s", i, e)
        pdf.close()
        if pages:
            return pages
    except ImportError as exc:
        raise IngestionError("OCR libraries not installed. Run: pip install easyocr pillow pymupdf") from exc
    except Exception as exc:
        raise IngestionError(f"OCR failed: {exc.__class__.__name__}: {exc}") from exc
    
    raise IngestionError("No extractable text found (the PDF may be blank or unreadable).")

def make_chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic id: identical unchanged content always yields the same id."""
    digest = hashlib.sha256(f"{source}|{page}|{index}|{text}".encode("utf-8")).hexdigest()
    return digest[:32]


def split_documents(pages: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: List[Document] = []
    for page in pages:
        for idx, piece in enumerate(splitter.split_documents([page])):
            piece.metadata["chunk_id"] = make_chunk_id(
                piece.metadata["source"], piece.metadata["page"], idx, piece.page_content
            )
            chunks.append(piece)
    return chunks


# ---------------------------------------------------------------------------- Pinecone
def _index_ready(pc: Pinecone, name: str) -> bool:
    status = pc.describe_index(name).status
    return bool(status["ready"])


def ensure_index(pc: Pinecone, settings: Settings, *, timeout: int = 300,
                 sleep: Callable[[float], None] = time.sleep) -> None:
    """Create the serverless index if needed and wait until it is ready."""
    name = settings.pinecone_index_name
    if pc.has_index(name):
        dim = pc.describe_index(name).dimension
        if dim != EMBEDDING_DIMENSION:
            raise IngestionError(f"Index {name!r} has dimension {dim}, expected {EMBEDDING_DIMENSION}.")
    else:
        logger.info("Creating Pinecone index %s", name)
        pc.create_index(
            name=name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    deadline = time.monotonic() + timeout
    while not _index_ready(pc, name):
        if time.monotonic() > deadline:
            raise IngestionError(f"Index {name!r} was not ready after {timeout}s.")
        sleep(2)


def upsert_chunks(index, embeddings, chunks: List[Document], *, batch_size: int = UPSERT_BATCH_SIZE,
                  sleep: Callable[[float], None] = time.sleep) -> dict:
    """Embed and upsert only chunks whose ids are not already stored. Returns counts."""
    added = skipped = 0
    for batch in batched(chunks, batch_size):
        ids = [c.metadata["chunk_id"] for c in batch]
        existing = set(with_retries(lambda: index.fetch(ids=ids), sleep=sleep).vectors.keys())
        new = [c for c in batch if c.metadata["chunk_id"] not in existing]
        skipped += len(batch) - len(new)
        if not new:
            continue
        vectors = with_retries(lambda: embeddings.embed_documents([c.page_content for c in new]), sleep=sleep)
        records = [
            {
                "id": c.metadata["chunk_id"],
                "values": vec,
                "metadata": {
                    "text": c.page_content,
                    "source": c.metadata["source"],
                    "page": c.metadata["page"],
                    "chunk_id": c.metadata["chunk_id"],
                },
            }
            for c, vec in zip(new, vectors)
        ]
        with_retries(lambda: index.upsert(vectors=records), sleep=sleep)
        added += len(new)
    return {"added": added, "skipped": skipped, "total": len(chunks)}


def ingest(pdf_path: Path = DEFAULT_PDF_PATH, *, settings: Optional[Settings] = None,
           reset: bool = False) -> dict:
    settings = settings or get_settings()
    settings.require_credentials()
    pages = load_pages(pdf_path)
    chunks = split_documents(pages)
    logger.info("Extracted %d pages -> %d chunks", len(pages), len(chunks))

    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(pc, settings)
    index = pc.Index(settings.pinecone_index_name)
    if reset:
        index.delete(delete_all=True)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    result = upsert_chunks(index, embeddings, chunks)
    result["pages"] = len(pages)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="Path to the PDF.")
    parser.add_argument("--url", default=PDF_DRIVE_URL, help="Google Drive URL used if the PDF is missing.")
    parser.add_argument("--reset", action="store_true", help="Delete all vectors in the index first.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if not args.pdf.exists():
            logger.info("PDF not found locally; trying to download it...")
            download_pdf(args.url, args.pdf)
        print(ingest(args.pdf, reset=args.reset))
    except (IngestionError, ConfigurationError) as exc:
        raise SystemExit(f"Ingestion failed: {exc}") from exc


if __name__ == "__main__":
    main()

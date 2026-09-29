"""Ingestion pipeline: download the eBook, chunk it, embed it and upsert it into Pinecone.

Run:  python -m src.ingestion            (add --reset to rebuild the index from scratch)
"""

import argparse
import hashlib
import re
import time

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.models import get_embeddings


def download_pdf(force: bool = False) -> None:
    """Fetch the eBook from Google Drive unless it is already on disk."""
    if config.PDF_PATH.exists() and not force:
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading eBook to {config.PDF_PATH} ...")
    resp = requests.get(config.PDF_DOWNLOAD_URL, timeout=120)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError("Download did not return a PDF. Download it manually into data/.")
    config.PDF_PATH.write_bytes(resp.content)


def clean_text(text: str) -> str:
    """Normalise whitespace left behind by PDF extraction."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages() -> list[Document]:
    """Load the PDF page by page, dropping empty pages (covers, full-page images)."""
    pages = PyPDFLoader(str(config.PDF_PATH)).load()
    docs = []
    for page in pages:
        text = clean_text(page.page_content)
        if len(text) < 30:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": config.PDF_PATH.name,
                    "page_number": int(page.metadata.get("page", 0)) + 1,
                },
            )
        )
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks; page metadata is kept on every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [
        c for c in splitter.split_documents(docs)
        if len(c.page_content.strip()) >= config.MIN_CHUNK_CHARS
    ]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic ID so re-running ingestion overwrites instead of duplicating."""
    raw = f"{chunk.metadata['page_number']}:{chunk.metadata['chunk_index']}:{chunk.page_content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def ensure_index(pc: Pinecone, reset: bool = False) -> None:
    """Create the serverless cosine index (1536 dims) if needed and wait until it is ready."""
    name = config.PINECONE_INDEX_NAME
    existing = {idx["name"] for idx in pc.list_indexes()}

    if reset and name in existing:
        print(f"Deleting existing index '{name}' ...")
        pc.delete_index(name)
        existing.discard(name)

    if name not in existing:
        print(f"Creating index '{name}' (dim={config.EMBEDDING_DIM}, metric=cosine) ...")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )

    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)


def run_ingestion(reset: bool = False) -> int:
    config.require_keys(config.PROVIDER_KEY, "PINECONE_API_KEY")
    download_pdf()

    pages = load_pages()
    chunks = chunk_documents(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, reset=reset)

    embeddings = get_embeddings()
    store = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=embeddings)
    # The chunk text is stored under the "text" metadata key, next to page_number and source.
    # Upload in batches; the Gemini free tier allows 100 embedding requests per minute.
    batch, pause = (50, 35) if config.LLM_PROVIDER == "gemini" else (100, 0)
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        store.add_documents(part, ids=[chunk_id(c) for c in part])
        print(f"  upserted {start + len(part)}/{len(chunks)}")
        if pause and start + batch < len(chunks):
            time.sleep(pause)

    print(f"Upserted {len(chunks)} vectors into '{config.PINECONE_INDEX_NAME}'.")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the index")
    args = parser.parse_args()
    run_ingestion(reset=args.reset)

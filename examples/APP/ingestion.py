# ingestion.py
import os
import re
import pathlib
from typing import List

from dotenv import load_dotenv
load_dotenv()
import uuid
from docling.document_converter import DocumentConverter

# LightRAG imports
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Import the background runner
from async_runner import run_coro_threadsafe, start_background_loop

# Files LightRAG writes; useful when resetting a workdir
_LR_FILES: List[str] = [
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
]

# Expose the supported modes in one place
QUERY_MODES = ["naive", "local", "global", "hybrid", "mix"]


def clear_workdir_files(working_dir: str):
    """Delete LightRAG output files so a fresh ingest can occur."""
    os.makedirs(working_dir, exist_ok=True)
    for fname in _LR_FILES:
        fpath = os.path.join(working_dir, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass


async def _abuild_rag(working_dir: str, chunk_token_size: int, chunk_overlap_token_size: int) -> LightRAG:
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def _ainsert_text(rag: LightRAG, text: str):
    await rag.ainsert(text)


def build_rag(working_dir: str, chunk_token_size: int = 200, chunk_overlap_token_size: int = 40) -> LightRAG:
    """
    Synchronously build a LightRAG instance — but schedule the build on the shared background loop
    so that all asyncio primitives inside LightRAG are bound to the same event loop used later for queries.
    """
    start_background_loop()  # ensure loop is running
    return run_coro_threadsafe(_abuild_rag(working_dir, chunk_token_size, chunk_overlap_token_size))


def insert_text_into_rag(rag: LightRAG, text: str):
    """Sync wrapper to insert text into LightRAG using the shared background loop."""
    return run_coro_threadsafe(_ainsert_text(rag, text))


def pdf_bytes_to_text(pdf_bytes: bytes, filename: str | None = None) -> str:
    """
    Convert PDF bytes to plain text using Docling and save the original PDF and the extracted text
    into .tmp_docling. If filename is provided, use that filename (preserving original name).
    This avoids writing a fixed upload.pdf/upload.txt pair.
    Returns the extracted text.
    """
    tmp_dir = working_dir_for_tmp()

    # Use provided filename if present; otherwise generate a unique name.
    if filename:
        # sanitize filename a little (remove path parts)
        base_name = os.path.basename(filename)
    else:
        base_name = f"doc_{uuid.uuid4().hex}.pdf"

    # Ensure we have a .pdf extension
    if not base_name.lower().endswith(".pdf"):
        base_name = base_name + ".pdf"

    pdf_path = pathlib.Path(tmp_dir) / base_name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the PDF bytes directly to the per-file path
    pdf_path.write_bytes(pdf_bytes)

    # Convert with Docling using the real file path (Docling expects a path)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    text = result.document.export_to_text()

    # Normalize whitespace a bit
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Save the extracted text next to the PDF with the same basename
    out_txt = pdf_path.with_suffix(".txt")
    out_txt.write_text(text, encoding="utf-8")

    return text


def working_dir_for_tmp() -> str:
    """
    Ensures a place exists to write conversion temp files (.tmp_docling under project root).
    """
    d = os.path.join(os.getcwd(), ".tmp_docling")
    os.makedirs(d, exist_ok=True)
    return d

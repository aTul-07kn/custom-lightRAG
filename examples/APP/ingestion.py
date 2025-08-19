# ingestion.py
import os
import re
import pathlib
from typing import List

from dotenv import load_dotenv
load_dotenv()

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


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """
    Convert PDF bytes to plain text using Docling.
    Saves temporary pdf under .tmp_docling and returns extracted text.
    """
    tmp_dir = working_dir_for_tmp()
    tmp_path = pathlib.Path(tmp_dir) / "upload.pdf"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(pdf_bytes)

    converter = DocumentConverter()
    result = converter.convert(str(tmp_path))
    text = result.document.export_to_text()

    # Normalize whitespace a bit
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Save the .txt alongside so user can inspect
    out_txt = tmp_path.with_suffix(".txt")
    out_txt.write_text(text, encoding="utf-8")

    return text


def working_dir_for_tmp() -> str:
    """
    Ensures a place exists to write conversion temp files (.tmp_docling under project root).
    """
    d = os.path.join(os.getcwd(), ".tmp_docling")
    os.makedirs(d, exist_ok=True)
    return d

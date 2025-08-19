# app.py (full file; replace your current app.py with this)
import os
import time
import shutil
import logging
import asyncio
import streamlit as st
from dotenv import load_dotenv

from ingestion import (
    build_rag,
    clear_workdir_files,
    pdf_bytes_to_text,
    insert_text_into_rag,
    QUERY_MODES,
)

# Runner helper (we need run_coro_threadsafe for queries)
from async_runner import start_background_loop, run_coro_threadsafe

# ---------------------
# Logging -> terminal only
# ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------
# Page config & styling
# ---------------------
st.set_page_config(page_title="LightRAG PDF QA", layout="wide")

load_dotenv()

# Ensure background loop is started early (idempotent)
start_background_loop()

# ---------------------
# App state
# ---------------------
if "rag" not in st.session_state:
    st.session_state.rag = None
if "workdir" not in st.session_state:
    st.session_state.workdir = os.path.abspath("TEMP")
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# Simple helper: Detect if a LightRAG workdir already contains storage files
STORAGE_FILES = [
    "graph_chunk_entity_relation.graphml",
    "kv_store_doc_status.json",
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
]


def storages_exist(workdir: str) -> bool:
    for f in STORAGE_FILES:
        if os.path.exists(os.path.join(workdir, f)):
            return True
    return False


def remove_and_recreate_dirs(workdir: str):
    """
    Remove the workdir (TEMP) and .tmp_docling directories if they exist, then recreate them empty.
    """
    # Remove TEMP (workdir)
    try:
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
            logger.info("Deleted existing workdir: %s", workdir)
    except Exception as e:
        logger.warning("Failed to delete workdir %s: %s", workdir, e)

    # Remove .tmp_docling under project root
    tmp_docling = os.path.join(os.getcwd(), ".tmp_docling")
    try:
        if os.path.exists(tmp_docling):
            shutil.rmtree(tmp_docling)
            logger.info("Deleted existing .tmp_docling: %s", tmp_docling)
    except Exception as e:
        logger.warning("Failed to delete .tmp_docling %s: %s", tmp_docling, e)

    # Recreate fresh directories
    try:
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(tmp_docling, exist_ok=True)
        logger.info("Recreated directories: %s and %s", workdir, tmp_docling)
    except Exception as e:
        logger.error("Failed to recreate directories: %s", e)
        raise


# Settings expander
with st.expander("⚙️ Settings", expanded=False):
    chunk_size = st.number_input("Chunk token size", min_value=50, max_value=1000, value=200, step=10)
    chunk_overlap = st.number_input("Chunk overlap token size", min_value=0, max_value=400, value=40, step=5)
    st.caption("These are passed to LightRAG during initialization.")

# ==========================
# 1) Ingestion section
# ==========================
st.header("1) Ingest PDF(s) into Knowledge Base")
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Single full-width button for processing (reset button removed per request)
process_btn = st.button("Process Uploaded PDFs", disabled=(not uploaded_files))

if process_btn and uploaded_files:
    # Delete existing TEMP and .tmp_docling, reset state, create fresh dirs
    logger.info("New upload detected — removing existing TEMP and .tmp_docling and recreating fresh directories.")
    remove_and_recreate_dirs(st.session_state.workdir)

    # Clear any LightRAG references in session state (fresh run)
    st.session_state.rag = None
    st.session_state.ingested_files = set()

    # Build RAG (this uses the background loop under the hood)
    logger.info("Initializing RAG for ingestion...")
    st.session_state.rag = build_rag(
        working_dir=st.session_state.workdir,
        chunk_token_size=chunk_size,
        chunk_overlap_token_size=chunk_overlap,
    )

    # Ensure tmp_docling path
    tmp_docling_dir = os.path.join(os.getcwd(), ".tmp_docling")
    os.makedirs(tmp_docling_dir, exist_ok=True)

    # Ingest each uploaded PDF; pdf_bytes_to_text now saves per-file pdf & txt into .tmp_docling
    for pdf_file in uploaded_files:
        with st.spinner(f"Ingesting `{pdf_file.name}`..."):
            try:
                logger.info(f"Processing `{pdf_file.name}`")
                # Read bytes once
                pdf_bytes = pdf_file.read()
                if not pdf_bytes:
                    raise RuntimeError(f"No bytes read from uploaded file {pdf_file.name}")

                # Convert to text and save both the per-file pdf and per-file txt inside .tmp_docling
                text = pdf_bytes_to_text(pdf_bytes, filename=pdf_file.name)

                # Insert text into RAG
                insert_text_into_rag(st.session_state.rag, text)
                st.session_state.ingested_files.add(pdf_file.name)
                logger.info(f"`{pdf_file.name}` ingested into KB.")
            except Exception as e:
                logger.exception("Failed to ingest %s: %s", pdf_file.name, e)
                st.error(f"Failed to ingest `{pdf_file.name}`: {e}")

    st.success("PDF ingestion complete. You may now ask questions.")

# ==========================
# 2) Query section (always visible)
# ==========================
st.header("2) Ask a question (uses current KB)")
if st.session_state.rag is None and not storages_exist(st.session_state.workdir):
    st.warning("No documents have been ingested yet and no on-disk KB was found. Upload PDFs")
else:
    # Inform if we found on-disk KB but haven't initialized rag yet
    if st.session_state.rag is None and storages_exist(st.session_state.workdir):
        st.info("Found an existing KB on disk. The RAG will be initialized automatically when you run the query.")

query = st.text_area("Enter your question", height=140)

mode = st.selectbox("Query mode", options=QUERY_MODES + ["all"], index=0)

# Enable Run button when the user typed a query and either we have an initialized rag OR storages exist (so we can lazy-init)
run_btn_disabled = (not query.strip()) or (st.session_state.rag is None and not storages_exist(st.session_state.workdir))
run_btn = st.button("Run query", disabled=run_btn_disabled)

def run_one_mode(rag, q, mode_name):
    from lightrag import QueryParam
    logger.info(f"Running mode: {mode_name}")
    start = time.perf_counter()

    # Use thread-safe runner to schedule coroutine on the shared loop
    result = run_coro_threadsafe(rag.aquery(q, param=QueryParam(mode=mode_name)))
    dur = (time.perf_counter() - start) * 1000.0
    logger.info(f"Mode {mode_name} done in {dur:.1f} ms")
    return result, dur

if run_btn:
    # If rag is not initialized but storages exist on disk, create it now (lazily).
    if st.session_state.rag is None:
        logger.info("Lazy-initializing RAG from existing storage...")
        st.session_state.rag = build_rag(
            working_dir=st.session_state.workdir,
            chunk_token_size=chunk_size,
            chunk_overlap_token_size=chunk_overlap,
        )
        logger.info("RAG initialized.")

    if mode == "all":
        st.subheader("Results by mode")
        for m in QUERY_MODES:
            with st.spinner(f"Running {m}..."):
                out, ms = run_one_mode(st.session_state.rag, query, m)
            with st.expander(f"Mode: {m} — {ms:.1f} ms", expanded=(m == "naive")):
                st.write(out)
    else:
        with st.spinner(f"Running {mode}..."):
            out, ms = run_one_mode(st.session_state.rag, query, mode)
        st.subheader(f"Answer ({mode}) — {ms:.1f} ms")
        st.write(out)

# Optional: list ingested files (full width)
if st.session_state.ingested_files:
    st.markdown("**Ingested files:** " + ", ".join(sorted(st.session_state.ingested_files)))
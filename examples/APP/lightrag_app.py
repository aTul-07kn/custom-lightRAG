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
    clear_workdir_files,      # still imported but not used after removing reset button
    pdf_bytes_to_text,
    insert_text_into_rag,
    QUERY_MODES,
)
from async_runner import start_background_loop, run_coro_threadsafe

# ---------------------
# Logging -> terminal only
# ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------
# Page config & startup
# ---------------------
st.set_page_config(page_title="Knowledge Graph Demo", layout="wide")
load_dotenv()
start_background_loop()

# ---------------------
# Session state
# ---------------------
if "rag" not in st.session_state:
    st.session_state.rag = None
if "workdir" not in st.session_state:
    st.session_state.workdir = os.path.abspath("TEMP")
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

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
    return any(os.path.exists(os.path.join(workdir, f)) for f in STORAGE_FILES)

# ---------------------
# Global UI styles (visual only)
# ---------------------
PRIMARY = "#1463FF"
TEXT    = "#0F172A"
BORDER  = "#E2E8F0"
SOFTBG  = "#F8FAFC"

st.markdown(
    f"""
    <style>
      /* Base */
      html, body, [class*="css"] {{
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial;
        color: {TEXT};
      }}

      /* H2 headers with built-in line (no extra sibling div needed) */
      h2.with-line {{
        font-weight: 800;
        margin: 4px 0 10px 0;   /* tight */
        position: relative;
        padding-bottom: 8px;    /* space for the line */
      }}
      h2.with-line::after {{
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 2px;
        background: {BORDER};
      }}

      /* Aggressively remove any stray "oval input" that some themes insert after headers */
      h2 + div input,
      h2 + div textarea,
      h2 + div [role="textbox"],
      h2 + div [data-baseweb="input"],
      h2 + div [data-baseweb="select"],
      h2 + div .stTextInput,
      h2 + div .stTextArea {{
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        background: transparent !important;
      }}
      h2 + div {{
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        background: transparent !important;
      }}

      /* Cards (compact) */
      .card {{
        border: 1px solid {BORDER};
        background: #fff;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
      }}
      .soft {{
        background: {SOFTBG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 6px;
      }}

      /* Primary buttons */
      .stButton > button {{
        border-radius: 9999px !important;
        padding: 9px 14px !important;
        font-weight: 700 !important;
        border: 1px solid {PRIMARY} !important;
        background: {PRIMARY} !important;
        color: #fff !important;
        margin-top: 2px !important;
      }}
      .stButton > button:disabled {{
        opacity: 0.5 !important;
        cursor: not-allowed !important;
      }}

      /* Labels: bigger & bold for the three you called out */
      label {{
        font-size: 16px !important;
        font-weight: 700 !important;
        color: {TEXT} !important;
        margin-bottom: 2px !important;
      }}

      /* Inputs: rounded + tidy */
      .stTextArea textarea,
      .stNumberInput input,
      .stSelectbox [data-baseweb="select"] > div,
      .stFileUploader div[data-testid="stFileUploaderDropzone"] {{
        border-radius: 12px !important;
        border-color: {BORDER} !important;
      }}

      /* Reduce global vertical gaps */
      .block-container {{ padding-top: 20px !important; }}
      .element-container {{ margin-bottom: 8px !important; }}

      /* File uploader header spacing */
      div[data-testid="stFileUploaderDropzone"] > div:first-child {{
        padding: 10px 14px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# 1) Ingestion section (same controls, NO reset button)
# ==========================
st.markdown('<h2 class="with-line">Knowledge Fabric</h2>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Document", type=["pdf"], accept_multiple_files=True)

btn_col_left, _spacer = st.columns([1, 1])
with btn_col_left:
    process_btn = st.button("Ingest Uploaded Documents", disabled=(not uploaded_files))

if process_btn and uploaded_files:
    os.makedirs(st.session_state.workdir, exist_ok=True)
    if st.session_state.rag is None:
        logger.info("Initializing RAG for the first time...")
        # These values are still driven by Settings at the end when you ingest after changing them.
        # If you want them live-threaded here immediately, say so and I'll wire through.
        chunk_size_default = 200
        chunk_overlap_default = 40
        st.session_state.rag = build_rag(
            working_dir=st.session_state.workdir,
            chunk_token_size=chunk_size_default,
            chunk_overlap_token_size=chunk_overlap_default,
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

    st.success("Document ingestion complete and Knowledge Graph is ready.")

# ==========================
# 2) Query section
# ==========================
st.markdown('<h2 class="with-line">Query Knowledge Fabric</h2>', unsafe_allow_html=True)

if st.session_state.rag is None and not storages_exist(st.session_state.workdir):
    st.warning("No documents exist in Knowledge Fabric")
else:
    if st.session_state.rag is None and storages_exist(st.session_state.workdir):
        st.info("Knowledge Graph Found.")

query = st.text_area("Enter Query", height=140)
# mode = st.selectbox("Query mode", options=QUERY_MODES + ["all"], index=0)
MODE_OPTIONS = QUERY_MODES + ["all"]
DISPLAY_MAP = {
    "naive": "Traditional",
    "local": "Knowledge Graph - Local",
    "global": "Knowledge Graph - Global",
    "hybrid": "Knowledge Graph - Hybrid",
    "mix": "Knowledge Graph - Mix",
    "all": "All",
}
mode = st.selectbox(
    "Query mode",
    options=MODE_OPTIONS,
    index=0,
    format_func=lambda v: DISPLAY_MAP.get(v, v),
)

run_btn_disabled = (not query.strip()) or (st.session_state.rag is None and not storages_exist(st.session_state.workdir))
run_btn = st.button("Run query", disabled=run_btn_disabled)

def run_one_mode(rag, q, mode_name):
    from lightrag import QueryParam
    logger.info(f"Running mode: {mode_name}")
    start = time.perf_counter()
    result = run_coro_threadsafe(rag.aquery(q, param=QueryParam(mode=mode_name)))
    dur = (time.perf_counter() - start) * 1000.0
    logger.info(f"Mode {mode_name} done in {dur:.1f} ms")
    return result, dur

if run_btn:
    if st.session_state.rag is None:
        logger.info("Lazy-initializing RAG from existing storage...")
        # Same defaults; can be wired to Settings if you want live control before ingest.
        chunk_size_default = 200
        chunk_overlap_default = 40
        st.session_state.rag = build_rag(
            working_dir=st.session_state.workdir,
            chunk_token_size=chunk_size_default,
            chunk_overlap_token_size=chunk_overlap_default,
        )
        logger.info("RAG initialized.")

    if mode == "all":
        st.subheader("Results by Mode")
        for m in QUERY_MODES:
            display_name = DISPLAY_MAP.get(m, m)
            # spinner uses friendly name
            with st.spinner(f"Running {display_name}..."):
                out, ms = run_one_mode(st.session_state.rag, query, m)
            # expander title uses friendly name; keep naive expanded by default
            with st.expander(f"{display_name} — {ms:.1f} ms", expanded=(m == "naive")):
                st.write(out)
    else:
        display_name = DISPLAY_MAP.get(mode, mode)
        with st.spinner(f"Running {display_name}..."):
            out, ms = run_one_mode(st.session_state.rag, query, mode)
        st.subheader(f"Answer ({display_name}) — {ms:.1f} ms")
        st.write(out)

# Optional: list ingested files
if st.session_state.ingested_files:
    st.markdown(
        f'<div class="soft"><strong>Ingested documents:</strong> {", ".join(sorted(st.session_state.ingested_files))}</div>',
        unsafe_allow_html=True,
    )

# ==========================
# 3) Settings (kept at the end)
# ==========================
with st.expander("⚙️ Settings", expanded=False):
    st.caption("These are passed to LightRAG during initialization.")
    chunk_size = st.number_input("Chunk token size", min_value=50, max_value=1000, value=200, step=10)
    chunk_overlap = st.number_input("Chunk overlap token size", min_value=0, max_value=400, value=40, step=5)
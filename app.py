import os
import pathlib
from pathlib import Path
from typing import List, Tuple

import logging
from logging.handlers import RotatingFileHandler

import gradio as gr
import pandas as pd

import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS
from markitdown import MarkItDown


# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / ".lotus_index"
ALLOWED_SUFFIXES = {
    ".pdf", ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx", ".csv",
    ".txt", ".md", ".html", ".xml", ".json",
    ".eml", ".msg", ".epub", ".zip",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".mp3", ".wav", ".m4a", ".ogg",
}
CONVERTED_DIR = DATA_DIR / "converted_md"
LOG_DIR = DATA_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
CONVERTED_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def _configure_logging() -> logging.Logger:
    env_log_level = os.environ.get("LOG_LEVEL", "").upper().strip()
    debug_env = os.environ.get("DEBUG", "false").lower() == "true"
    level_name = env_log_level or ("DEBUG" if debug_env else "INFO")
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("docextract")
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplication on reloads
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating file handler (5MB x 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logging configured: level=%s file=%s", level_name or ("DEBUG" if debug_env else "INFO"), LOG_FILE)
    return logger


LOGGER = _configure_logging()


def _configure_lotus() -> None:
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    LOGGER.info("Configuring LOTUS provider=%s", provider)

    # Allow overriding model names via environment variables
    llm_model = os.environ.get("LLM_MODEL", "gpt-4.1-nano").strip()
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    LOGGER.debug("Models selected llm_model=%s embedding_model=%s", llm_model, embedding_model)

    if provider == "ollama":
        # Use Ollama's OpenAI-compatible endpoint. Assume local default port.
        # If user provided a custom base, respect it.
        openai_base = os.environ.get("OPENAI_API_BASE", "").strip()
        if not openai_base:
            os.environ["OPENAI_API_BASE"] = os.environ.get(
                "OLLAMA_OPENAI_BASE", "http://localhost:11434/v1"
            )

        # Ollama doesn't require a real API key; set a placeholder if absent
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "ollama"

        # Reasonable defaults for local models if not overridden
        if "LLM_MODEL" not in os.environ:
            llm_model = "llama3.2"
        if "EMBEDDING_MODEL" not in os.environ:
            embedding_model = "nomic-embed-text"

        try:
            lotus.settings.configure(
                lm=LM(api_key=api_key, model=llm_model),
                rm=LiteLLMRM(model=embedding_model),
                vs=FaissVS(),
            )
            LOGGER.info("LOTUS configured for Ollama (OpenAI-compatible) base=%s", os.environ.get("OPENAI_API_BASE"))
        except Exception:
            LOGGER.exception("Failed configuring LOTUS for Ollama")
            raise
    else:
        # Default: OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required to run this app (unless LLM_PROVIDER=ollama)."
            )
        # Ensure we don't accidentally point to a non-OpenAI base when using OpenAI
        if os.environ.get("LLM_PROVIDER", "openai").strip().lower() == "openai":
            # If someone previously set OPENAI_API_BASE for Ollama, ignore it here
            if os.environ.get("OPENAI_API_BASE", "").strip().startswith("http://localhost:11434"):
                os.environ.pop("OPENAI_API_BASE", None)
        try:
            lotus.settings.configure(
                lm=LM(api_key=api_key, model=llm_model),
                rm=LiteLLMRM(model=embedding_model),
                vs=FaissVS(),
            )
            LOGGER.info("LOTUS configured for OpenAI model=%s embed=%s", llm_model, embedding_model)
        except Exception:
            LOGGER.exception("Failed configuring LOTUS for OpenAI")
            raise


_configure_lotus()


# -----------------------------
# File utilities
# -----------------------------
def list_data_files() -> List[str]:
    LOGGER.debug("Listing data files in %s", DATA_DIR)
    files = [
        f.name
        for f in sorted(DATA_DIR.iterdir(), key=lambda p: p.name.lower())
        if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]
    LOGGER.info("Found %d supported files", len(files))
    return files


def save_uploaded_files(files: List[gr.File]) -> str:
    if not files:
        LOGGER.warning("Upload invoked with empty file list")
        return "No files provided."
    saved = 0
    for file_obj in files:
        # gradio may pass TemporaryFile with .name or dict with 'name'
        name = getattr(file_obj, "name", None) or getattr(file_obj, "orig_name", None)
        if not name:
            # fall back to basename of temp path
            try:
                name = Path(file_obj).name  # type: ignore[arg-type]
            except Exception:
                name = None
        if not name:
            LOGGER.error("Could not determine filename for uploaded object: %r", file_obj)
            continue
        dest = DATA_DIR / Path(name).name
        # Some uploaders provide .read(); some provide temp path
        try:
            # If file_obj is a tempfile-like
            content = file_obj.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved += 1
            LOGGER.info("Saved uploaded file to %s (via .read())", dest)
        except Exception:
            LOGGER.debug("Direct read failed for %s, attempting temp path copy", name, exc_info=True)
            # Try to copy from temp path
            try:
                temp_path = Path(getattr(file_obj, "name", str(file_obj)))
                dest.write_bytes(Path(temp_path).read_bytes())
                saved += 1
                LOGGER.info("Copied uploaded file from temp path %s to %s", temp_path, dest)
            except Exception:
                LOGGER.exception("Failed to persist uploaded file %s", name)
                continue
    msg = f"Uploaded {saved} file(s)."
    LOGGER.debug("Upload result: %s", msg)
    return msg


def remove_file(file_name: str) -> str:
    if not file_name:
        LOGGER.warning("Remove requested with no file selected")
        return "No file selected."
    target = DATA_DIR / file_name
    if target.exists():
        try:
            target.unlink()
            # Also remove converted Markdown counterpart if present
            converted = CONVERTED_DIR / (Path(file_name).stem + ".md")
            if converted.exists():
                try:
                    converted.unlink()
                except Exception:
                    pass
            LOGGER.info("Removed file %s and its converted counterpart if existed", file_name)
            return f"Removed {file_name}."
        except Exception as e:
            LOGGER.exception("Failed to remove file %s", file_name)
            return f"Failed to remove {file_name}: {e}"
    LOGGER.warning("Requested removal of non-existent file %s", file_name)
    return f"File {file_name} not found."


# -----------------------------
# Conversion, Ingestion and QA
# -----------------------------
def convert_all_to_markdown() -> Tuple[int, int]:
    """Convert all supported files in data/ to Markdown under data/converted_md/.

    Returns a tuple of (num_converted, num_failed).
    """
    LOGGER.info("Starting conversion of all files to Markdown")
    md_converter = MarkItDown(enable_plugins=False)
    converted = 0
    failed = 0
    
    files_to_convert = [
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.parent != CONVERTED_DIR and f.suffix.lower() in ALLOWED_SUFFIXES
    ]
    LOGGER.debug("Found %d files to convert: %s", len(files_to_convert), [f.name for f in files_to_convert])
    
    for f in files_to_convert:
        out_path = CONVERTED_DIR / (f.stem + ".md")
        LOGGER.debug("Converting %s -> %s", f.name, out_path.name)
        try:
            result = md_converter.convert(str(f))
            text = getattr(result, "text_content", None) or ""
            if not text.strip():
                LOGGER.warning("MarkItDown produced empty text for %s", f.name)
            else:
                LOGGER.debug("MarkItDown extracted %d chars from %s", len(text), f.name)
            out_path.write_text(text, encoding="utf-8")
            converted += 1
            LOGGER.info("Successfully converted %s (%d chars)", f.name, len(text))
        except Exception as e:
            failed += 1
            LOGGER.exception("Failed to convert %s: %s", f.name, e)
            continue
    
    LOGGER.info("Conversion complete: %d converted, %d failed", converted, failed)
    return converted, failed


def build_pages_dataframe() -> pd.DataFrame:
    LOGGER.info("Building pages dataframe")
    # Ensure all current docs are converted to Markdown first
    try:
        converted, failed = convert_all_to_markdown()
        LOGGER.debug("Pre-conversion results: %d converted, %d failed", converted, failed)
    except Exception as e:
        LOGGER.exception("Conversion step failed: %s", e)

    md_files = list(sorted(CONVERTED_DIR.glob("*.md"), key=lambda p: p.name.lower()))
    LOGGER.debug("Found %d markdown files in %s", len(md_files), CONVERTED_DIR)
    
    rows = []
    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                LOGGER.warning("Markdown file %s is empty or whitespace-only", f.name)
            else:
                LOGGER.debug("Read %d chars from %s", len(text), f.name)
        except Exception as e:
            LOGGER.exception("Failed to read markdown file %s: %s", f.name, e)
            text = ""
        rows.append({
            "content": text,
            "path": str(f),
            "page": "1",
        })

    if not rows:
        LOGGER.warning("No rows built from markdown files - dataframe will be empty")
        return pd.DataFrame(columns=["content", "path", "page"])
    
    df = pd.DataFrame(rows, columns=["content", "path", "page"])
    LOGGER.info("Built dataframe with %d rows", len(df))
    return df


def _extract_citation_fields(df_row: pd.Series) -> Tuple[str, str]:
    # Try common field names for path and page
    path = None
    for key in [
        "path",
        "file_path",
        "source_path",
        "document_path",
    ]:
        if key in df_row and pd.notna(df_row[key]):
            path = str(df_row[key])
            break
    if path is None and "metadata" in df_row and isinstance(df_row["metadata"], dict):
        path = str(df_row["metadata"].get("path"))
    if path is None:
        path = "unknown"

    page_val = None
    for key in ["page", "page_num", "page_number"]:
        if key in df_row and pd.notna(df_row[key]):
            page_val = str(df_row[key])
            break
    if page_val is None:
        page_val = "?"

    return path, page_val


def answer_question(question: str, top_k: int = 5) -> Tuple[str, pd.DataFrame, List[str]]:
    LOGGER.info("Processing question: %s (top_k=%s)", question, top_k)
    
    if not question or not question.strip():
        LOGGER.warning("Empty question provided")
        return "Please enter a question.", pd.DataFrame(), list_data_files()

    df = build_pages_dataframe()
    if len(df) == 0:
        LOGGER.error("No documents available - dataframe is empty")
        return "No documents found in data/. Upload or place files there and try again.", pd.DataFrame(), list_data_files()

    LOGGER.debug("Dataframe columns: %s, shape: %s", list(df.columns), df.shape)

    # Ensure we have the expected content column
    if "content" not in df.columns:
        LOGGER.warning("'content' column missing, searching for alternatives in: %s", list(df.columns))
        # Try to find a reasonable text column name
        text_col = None
        for cand in ["text", "chunk", "page_text", "body"]:
            if cand in df.columns:
                text_col = cand
                break
        if text_col is None:
            LOGGER.error("No suitable text column found in dataframe")
            return "Could not locate text content in documents.", pd.DataFrame(), list_data_files()
        LOGGER.info("Renaming column %s to 'content'", text_col)
        df = df.rename(columns={text_col: "content"})

    # Check content quality
    empty_content_count = df["content"].str.strip().eq("").sum()
    if empty_content_count > 0:
        LOGGER.warning("%d out of %d documents have empty content", empty_content_count, len(df))

    # Build or load a semantic index and retrieve top passages
    LOGGER.info("Building semantic index for %d documents", len(df))
    try:
        df = df.sem_index("content", str(INDEX_DIR))
        LOGGER.info("Semantic indexing completed successfully")
    except Exception as e:
        LOGGER.exception("Semantic indexing failed: %s", e)
        # If indexing fails, proceed without persistence
        pass

    try:
        top_k = max(1, min(int(top_k), 20))
    except Exception:
        LOGGER.warning("Invalid top_k value, defaulting to 5")
        top_k = 5

    LOGGER.info("Performing semantic search with top_k=%d", top_k)
    try:
        retrieved = df.sem_search("content", question, K=top_k)
        LOGGER.info("Search completed, retrieved %d passages", len(retrieved))
    except Exception as e:
        LOGGER.exception("Semantic search failed: %s", e)
        return f"Search failed: {e}", pd.DataFrame(), list_data_files()

    if len(retrieved) == 0:
        LOGGER.warning("Search returned no relevant passages")
        return "No relevant passages found.", pd.DataFrame(), list_data_files()

    # Construct citations table and a small context preview
    LOGGER.debug("Building citations from %d retrieved passages", len(retrieved))
    previews = []
    for _, row in retrieved.iterrows():
        path, page = _extract_citation_fields(row)
        content = str(row.get("content", ""))
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        previews.append({
            "file": Path(path).name,
            "page": page,
            "snippet": snippet,
        })

    citations_df = pd.DataFrame(previews, columns=["file", "page", "snippet"])
    LOGGER.debug("Built citations table with %d entries", len(citations_df))

    # Ask LOTUS to produce a grounded answer using the retrieved content
    LOGGER.info("Generating answer using LOTUS aggregation")
    try:
        agg_prompt = (
            "You are a helpful assistant. Using the provided passages {content}, "
            "answer the user's question as precisely as possible. "
            "When you use information from a passage, include a citation like [n] where n refers to the passage index in the provided list. "
            "If you are unsure, say you don't know."
        )
        # Add a stable ordering index for references
        retrieved = retrieved.reset_index(drop=True).reset_index(names=["ref"]).assign(ref=lambda d: d["ref"] + 1)
        # We only need the LLM to write the answer; we don't need to persist its column on the DF
        answer_df = retrieved.sem_agg(
            agg_prompt + f"\n\nUser question: {question}",
            suffix="answer",
        )
        answer_text = str(answer_df.iloc[0].answer)
        LOGGER.info("Successfully generated answer (%d chars)", len(answer_text))
    except Exception as e:
        LOGGER.exception("Failed to generate answer: %s", e)
        answer_text = f"Failed to generate answer: {e}"

    # Add an explicit references section (file + page) to aid grounding
    refs_lines = [
        f"[{idx+1}] {row['file']} (page {row['page']})" for idx, row in citations_df.iterrows()
    ]
    references_md = "\n".join(refs_lines)
    final_md = f"{answer_text}\n\n---\n**References**\n\n{references_md}"

    LOGGER.info("Question processing complete")
    return final_md, citations_df, list_data_files()


# -----------------------------
# Log Viewer Utilities
# -----------------------------
def get_recent_logs(num_lines: int = 100) -> str:
    """Get recent log entries from the log file."""
    try:
        if not LOG_FILE.exists():
            return "Log file not found."
        
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Return the last num_lines
        recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
        return "".join(recent_lines)
    except Exception as e:
        return f"Error reading logs: {e}"


def clear_logs() -> str:
    """Clear the log file."""
    try:
        if LOG_FILE.exists():
            LOG_FILE.write_text("", encoding="utf-8")
            LOGGER.info("Log file cleared by user")
            return "Logs cleared successfully."
        return "Log file not found."
    except Exception as e:
        LOGGER.exception("Failed to clear logs")
        return f"Error clearing logs: {e}"


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Document Q&A (LOTUS)") as demo:
    gr.Markdown("""
    ### Document Q&A (LOTUS)
    - Add PDF/DOC/DOCX files to `data/` via upload or direct copy.
    - Ask questions; answers cite specific passages.
    - Check the **Logs** tab to troubleshoot issues.
    """)

    with gr.Tabs():
        with gr.TabItem("Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Manage documents**")
                    files_gallery = gr.Dropdown(
                        choices=list_data_files(), label="Files in data/", interactive=True
                    )
                    with gr.Row():
                        uploader = gr.Files(
                            label="Upload documents (PDF, Office, images, audio, etc.)",
                            file_count="multiple", type="filepath"
                        )
                        upload_btn = gr.Button("Upload")
                    with gr.Row():
                        remove_btn = gr.Button("Remove selected")
                        refresh_btn = gr.Button("Refresh list")
                    file_ops_status = gr.Markdown(visible=True)

                with gr.Column(scale=2):
                    gr.Markdown("**Ask a question**")
                    question = gr.Textbox(placeholder="Type your question here...", lines=2)
                    topk = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Top-K passages")
                    ask_btn = gr.Button("Ask")
                    answer = gr.Markdown()
                    contexts = gr.Dataframe(
                        headers=["file", "page", "snippet"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        label="Retrieved passages",
                    )

        with gr.TabItem("Logs"):
            gr.Markdown("**Application Logs** - Monitor file processing and search operations")
            with gr.Row():
                log_lines = gr.Slider(
                    minimum=50, maximum=500, step=50, value=100,
                    label="Number of recent log lines to show"
                )
                with gr.Column():
                    refresh_logs_btn = gr.Button("Refresh Logs")
                    clear_logs_btn = gr.Button("Clear Logs", variant="secondary")
            
            log_display = gr.Textbox(
                value=get_recent_logs(100),
                lines=25,
                max_lines=25,
                label="Recent Logs",
                interactive=False,
                show_copy_button=True
            )

    # Wire actions
    def _do_upload(files: List[gr.File]):
        msg = save_uploaded_files(files)
        return msg, gr.update(choices=list_data_files())

    upload_btn.click(_do_upload, inputs=[uploader], outputs=[file_ops_status, files_gallery])

    def _do_remove(selected: str):
        msg = remove_file(selected)
        return msg, gr.update(choices=list_data_files(), value=None)

    remove_btn.click(_do_remove, inputs=[files_gallery], outputs=[file_ops_status, files_gallery])

    def _do_refresh():
        return gr.update(choices=list_data_files())

    refresh_btn.click(_do_refresh, outputs=[files_gallery])

    def _do_ask(q: str, k: int):
        md, df, _ = answer_question(q, k)
        return md, df

    ask_btn.click(_do_ask, inputs=[question, topk], outputs=[answer, contexts])

    # Wire log viewer actions
    def _refresh_logs(num_lines: int):
        return get_recent_logs(num_lines)

    def _clear_logs():
        msg = clear_logs()
        return msg, ""

    refresh_logs_btn.click(_refresh_logs, inputs=[log_lines], outputs=[log_display])
    clear_logs_btn.click(_clear_logs, outputs=[file_ops_status, log_display])


if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    try:
        port = int(os.environ.get("PORT", "7860"))
    except Exception:
        port = 7860
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    LOGGER.info("Starting Document Q&A app on %s:%d (debug=%s)", host, port, debug)
    LOGGER.info("Data directory: %s", DATA_DIR)
    LOGGER.info("Log file: %s", LOG_FILE)
    
    demo.launch(share=False, debug=debug, server_name=host, server_port=port)



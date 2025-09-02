import os
import pathlib
from pathlib import Path
from typing import List, Tuple

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

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
CONVERTED_DIR.mkdir(exist_ok=True)


def _configure_lotus() -> None:
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

    # Allow overriding model names via environment variables
    llm_model = os.environ.get("LLM_MODEL", "gpt-4.1-nano").strip()
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip()

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

        lotus.settings.configure(
            lm=LM(api_key=api_key, model=llm_model),
            rm=LiteLLMRM(model=embedding_model),
            vs=FaissVS(),
        )
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

        lotus.settings.configure(
            lm=LM(api_key=api_key, model=llm_model),
            rm=LiteLLMRM(model=embedding_model),
            vs=FaissVS(),
        )


_configure_lotus()


# -----------------------------
# File utilities
# -----------------------------
def list_data_files() -> List[str]:
    files = [
        f.name
        for f in sorted(DATA_DIR.iterdir(), key=lambda p: p.name.lower())
        if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]
    return files


def save_uploaded_files(files: List[gr.File]) -> str:
    if not files:
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
            continue
        dest = DATA_DIR / Path(name).name
        # Some uploaders provide .read(); some provide temp path
        try:
            # If file_obj is a tempfile-like
            content = file_obj.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved += 1
        except Exception:
            # Try to copy from temp path
            try:
                temp_path = Path(getattr(file_obj, "name", str(file_obj)))
                dest.write_bytes(Path(temp_path).read_bytes())
                saved += 1
            except Exception:
                continue
    return f"Uploaded {saved} file(s)."


def remove_file(file_name: str) -> str:
    if not file_name:
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
            return f"Removed {file_name}."
        except Exception as e:
            return f"Failed to remove {file_name}: {e}"
    return f"File {file_name} not found."


# -----------------------------
# Conversion, Ingestion and QA
# -----------------------------
def convert_all_to_markdown() -> Tuple[int, int]:
    """Convert all supported files in data/ to Markdown under data/converted_md/.

    Returns a tuple of (num_converted, num_failed).
    """
    md_converter = MarkItDown(enable_plugins=False)
    converted = 0
    failed = 0
    for f in DATA_DIR.iterdir():
        if not f.is_file():
            continue
        if f.parent == CONVERTED_DIR:
            continue
        if f.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        out_path = CONVERTED_DIR / (f.stem + ".md")
        try:
            result = md_converter.convert(str(f))
            text = getattr(result, "text_content", None) or ""
            out_path.write_text(text, encoding="utf-8")
            converted += 1
        except Exception:
            failed += 1
            continue
    return converted, failed


def build_pages_dataframe() -> pd.DataFrame:
    # Ensure all current docs are converted to Markdown first
    try:
        convert_all_to_markdown()
    except Exception:
        pass

    rows = []
    for f in sorted(CONVERTED_DIR.glob("*.md"), key=lambda p: p.name.lower()):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        rows.append({
            "content": text,
            "path": str(f),
            "page": "1",
        })

    if not rows:
        return pd.DataFrame(columns=["content", "path", "page"])
    return pd.DataFrame(rows, columns=["content", "path", "page"])


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
    if not question or not question.strip():
        return "Please enter a question.", pd.DataFrame(), list_data_files()

    df = build_pages_dataframe()
    if len(df) == 0:
        return "No documents found in data/. Upload or place files there and try again.", pd.DataFrame(), list_data_files()

    # Ensure we have the expected content column
    if "content" not in df.columns:
        # Try to find a reasonable text column name
        text_col = None
        for cand in ["text", "chunk", "page_text", "body"]:
            if cand in df.columns:
                text_col = cand
                break
        if text_col is None:
            return "Could not locate text content in documents.", pd.DataFrame(), list_data_files()
        df = df.rename(columns={text_col: "content"})

    # Build or load a semantic index and retrieve top passages
    try:
        df = df.sem_index("content", str(INDEX_DIR))
    except Exception:
        # If indexing fails, proceed without persistence
        pass

    try:
        top_k = max(1, min(int(top_k), 20))
    except Exception:
        top_k = 5

    try:
        retrieved = df.sem_search("content", question, K=top_k)
    except Exception as e:
        return f"Search failed: {e}", pd.DataFrame(), list_data_files()

    if len(retrieved) == 0:
        return "No relevant passages found.", pd.DataFrame(), list_data_files()

    # Construct citations table and a small context preview
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

    # Ask LOTUS to produce a grounded answer using the retrieved content
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
    except Exception as e:
        answer_text = f"Failed to generate answer: {e}"

    # Add an explicit references section (file + page) to aid grounding
    refs_lines = [
        f"[{idx+1}] {row['file']} (page {row['page']})" for idx, row in citations_df.iterrows()
    ]
    references_md = "\n".join(refs_lines)
    final_md = f"{answer_text}\n\n---\n**References**\n\n{references_md}"

    return final_md, citations_df, list_data_files()


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Document Q&A (LOTUS)") as demo:
    gr.Markdown("""
    ### Document Q&A (LOTUS)
    - Add PDF/DOC/DOCX files to `data/` via upload or direct copy.
    - Ask questions; answers cite specific passages.
    """)

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


if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    try:
        port = int(os.environ.get("PORT", "7860"))
    except Exception:
        port = 7860
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    demo.launch(share=False, debug=debug, server_name=host, server_port=port)



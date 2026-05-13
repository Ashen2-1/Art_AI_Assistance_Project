################################################################
### Phase III - RAG Query Pipeline
### Model: llava-hf/llava-1.5-7b-hf  (4-bit quantized, RTX 5070)
###
### Three modes:
###   text   - Retrieved PDF chunks -> LLaVA answers (no image)
###   vlm    - Image + question -> LLaVA answers (no retrieval)
###   hybrid - Image + retrieved chunks + question -> LLaVA answers
###
### Usage (CLI):
###   python rag_query.py text   "What is the uncanny?"
###   python rag_query.py vlm    "Describe this artwork." artwork.jpg
###   python rag_query.py hybrid "How does this relate to Freud?" artwork.jpg
###
### Usage (as module):
###   from rag_query import query_text_rag, query_vlm_only, query_hybrid
################################################################

import sys
import textwrap
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

# embed_pipeline.py and db.py are in the same folder
sys.path.insert(0, str(Path(__file__).parent))
from embed_pipeline import get_embed_model
from db import search_chunks, get_stats

# ── Config ────────────────────────────────────────────────────
MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
TOP_K          = 3        # chunks retrieved per query
MAX_NEW_TOKENS = 512

# ── LLaVA (loaded once, kept in GPU memory) ───────────────────
_processor = None
_model     = None


def load_llava():
    """
    Load LLaVA 1.5-7B in 4-bit quantization onto the RTX 5070.
    Called automatically on first query. Subsequent queries reuse
    the cached model — no reload overhead.
    """
    global _processor, _model
    if _model is not None:
        return _processor, _model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for LLaVA inference.\n"
            "Run:  pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    print(f"[RAG] Loading LLaVA 1.5-7B (4-bit) on GPU ...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    _processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    _model     = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
        dtype=torch.float16,
    ).to("cuda")
    _model.eval()

    print("[RAG] LLaVA ready.")
    return _processor, _model


# ── Retrieval ─────────────────────────────────────────────────
def retrieve(
    question: str,
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
) -> List[dict]:
    """
    Embed the question with BGE and retrieve the top-k most
    semantically similar chunks from PostgreSQL + pgvector.

    Parameters
    ----------
    question      : user's research question
    top_k         : number of chunks to retrieve (default 3)
    source_filter : restrict retrieval to one PDF filename (optional)

    Returns
    -------
    List of dicts with keys: text, source, chunk_idx, score
    """
    stats = get_stats()
    if stats["total_chunks"] == 0:
        print("[RAG] PostgreSQL is empty -- ingest a PDF first.")
        print("      Run: python embed_pipeline.py <file.pdf>")
        return []

    embed_model = get_embed_model()
    q_vec       = embed_model.encode([question], normalize_embeddings=True)[0].tolist()

    rows = search_chunks(q_vec, top_k=top_k, source_filter=source_filter)

    return [
        {
            "text"      : row["content"],
            "source"    : row["source"],
            "chunk_idx" : row["chunk_index"],
            "score"     : round(float(row["score"]), 4),
        }
        for row in rows
    ]


# ── Generation ────────────────────────────────────────────────
SYSTEM_MSG = (
    "You are an expert AI research assistant specialising in art history.\n"
    "Answer ONLY based on the provided context and/or image when RAG context is supplied.\n"
    "Use the conversation history to understand follow-up questions.\n"
    "If the answer cannot be found in the context, say exactly: "
    "'I cannot find this in the provided sources.'\n"
    "Format answers clearly using short paragraphs.\n"
    "Use Markdown formatting.\n"
    "If using bullet points or numbered points, place each item on its own line.\n"
    "Avoid large dense walls of text.\n"
    "Be precise. When possible, mention which source supports your answer."
)

def format_chat_history(chat_history: list) -> str:
    if not chat_history:
        return "No previous conversation"

    lines = []

    for msg in chat_history[-8:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if not content:
            continue

        if role == "ai":
            role_label = "ASSISTANT"
        else:
            role_label = "USER"

        lines.append(f"{role_label} {content}")

    return "\n".join(lines) if lines else "No previous conversation"

def _generate(prompt: str, image: Optional[Image.Image] = None) -> str:
    """Run LLaVA inference. Pass image=None for text-only queries."""
    processor, model = load_llava()

    if image is not None:
        inputs = processor(
            text   = prompt,
            images = image,
            return_tensors = "pt",
        ).to("cuda")
    else:
        # Text-only: no <image> token in prompt, no image tensor
        inputs = processor(
            text           = prompt,
            return_tensors = "pt",
        ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens    = MAX_NEW_TOKENS,
            do_sample         = False,          # deterministic = more reliable for research
            repetition_penalty= 1.05,
        )

    full = processor.decode(output_ids[0], skip_special_tokens=True)

    # LLaVA echoes the full prompt — strip it, keep only the answer
    if "ASSISTANT:" in full:
        return full.split("ASSISTANT:")[-1].strip()
    return full.strip()


# ── MODE 1: Text RAG ──────────────────────────────────────────
def query_text_rag(
    question: str,
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
    chat_history: list = None,
) -> dict:
    """
    TEXT RAG MODE
    -------------
    Retrieve the most relevant chunks from your ingested PDFs,
    then ask LLaVA to answer strictly from those chunks.

    No image required. Best for:
      - Questions about uploaded PDF documents
      - Literature review and source synthesis
      - Finding specific passages or arguments

    Parameters
    ----------
    question      : research question
    top_k         : how many chunks to retrieve (default 3)
    source_filter : limit retrieval to one PDF (e.g. "Sigmund.pdf")
    """
    print(f"\n[RAG] Mode: TEXT RAG  |  question: {question!r}")

    chunks  = retrieve(question, top_k=top_k, source_filter=source_filter)
    if not chunks:
        return {"answer": "No documents found in DB. Please ingest a PDF first.",
                "chunks": [], "mode": "text_rag"}

    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_idx'] + 1}]\n{c['text']}"
        for c in chunks
    )

    history_text = format_chat_history(chat_history or [])

    prompt = (
        f"{SYSTEM_MSG}\n\n"
        "Use the conversation history to understand follow-up questions.\n"
        "However, final factual claims must be supported by the provided context.\n\n"
        f"Selected source file: {source_filter or 'None'}\n\n"
        f"<conversation_history>\n{history_text}\n</conversation_history>\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}\n\n"
        f"ASSISTANT:"
    )

    answer = _generate(prompt, image=None)
    return {"answer": answer, "chunks": chunks, "mode": "text_rag"}


def query_general(question: str, chat_history: list = None) -> dict:
    """
    GENERAL CHAT MODE
    -----------------
    Answer general questions without requiring PDF context.
    Best for greetings, explanations, coding questions, and basic help.
    """
    print(f"\n[RAG] Mode: GENERAL CHAT  |  question: {question!r}")

    history_text = format_chat_history(chat_history or [])

    prompt = (
        "You are a helpful AI assistant.\n"
        "Continue the conversation naturally.\n"
        "Use the previous conversation for context when relevant.\n"
        "If the user asks a follow-up question, connect it to the earlier topic.\n"
        "Answer clearly and helpfully.\n"
        "Use Markdown formatting.\n"
        "Use short paragraphs.\n"
        "Use bullet points or numbered lists when helpful.\n"
        "Put each list item on its own line.\n"
        "Avoid giant unbroken blocks of text.\n\n"
        f"<conversation_history>\n{history_text}\n</conversation_history>\n\n"
        f"USER: {question}\n\n"
        "ASSISTANT:"
    )

    answer = _generate(prompt, image=None)

    return {
        "answer": answer,
        "mode": "general_chat",
        "chunks": [],
    }


# ── MODE 2: VLM Only ──────────────────────────────────────────
def query_vlm_only(question: str, image_path: str) -> dict:
    """
    VLM-ONLY MODE
    -------------
    Pass an image directly to LLaVA. No text retrieval.

    Best for:
      - Visual analysis of artworks
      - Style, composition, iconography questions
      - Artwork identification and description
      - Comparing visual elements in the image
    """
    print(f"\n[RAG] Mode: VLM ONLY  |  image: {image_path}")

    image  = Image.open(image_path).convert("RGB")
    prompt = (
        f"USER: <image>\n"
        f"{SYSTEM_MSG}\n\n"
        f"Question: {question}\n\n"
        f"ASSISTANT:"
    )

    answer = _generate(prompt, image=image)
    return {"answer": answer, "mode": "vlm_only", "image": image_path}


# ── MODE 3: Hybrid ────────────────────────────────────────────
def query_hybrid(
    question: str,
    image_path: str,
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
) -> dict:
    """
    HYBRID MODE
    -----------
    Combine image + retrieved text chunks → LLaVA answers from both.

    LLaVA sees the artwork AND the relevant passages from your
    research documents at the same time — the most powerful mode.

    Best for:
      - "How does this painting relate to what Freud said about X?"
      - Connecting a specific artwork to documented art history
      - Cross-referencing visual evidence with textual sources
    """
    print(f"\n[RAG] Mode: HYBRID  |  image: {image_path}")

    chunks  = retrieve(question, top_k=top_k, source_filter=source_filter)
    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_idx'] + 1}]\n{c['text']}"
        for c in chunks
    ) if chunks else "No text context available."

    image  = Image.open(image_path).convert("RGB")
    prompt = (
        f"USER: <image>\n"
        f"{SYSTEM_MSG}\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}\n\n"
        f"ASSISTANT:"
    )

    answer = _generate(prompt, image=image)
    return {"answer": answer, "chunks": chunks, "mode": "hybrid", "image": image_path}


# ── Pretty print ──────────────────────────────────────────────
def print_result(result: dict):
    print("\n" + "=" * 65)
    print(f"  MODE: {result['mode'].upper().replace('_', ' ')}")
    print("=" * 65)

    print("\n ANSWER:\n")
    for line in result["answer"].splitlines():
        print(textwrap.fill(line, width=65) if line.strip() else "")

    if result.get("chunks"):
        print("\n RETRIEVED SOURCES:")
        for i, c in enumerate(result["chunks"]):
            print(f"\n  [{i + 1}] {c['source']}  "
                  f"chunk {c['chunk_idx'] + 1}  "
                  f"similarity={c['score']}")
            preview = c["text"][:150].replace("\n", " ").strip()
            print(f"      \"{preview}...\"")

    if result.get("image"):
        print(f"\n  Image: {result['image']}")

    print("\n" + "=" * 65)


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python rag_query.py text   <question>")
        print("  python rag_query.py vlm    <question>  <image_path>")
        print("  python rag_query.py hybrid <question>  <image_path>")
        print("\nExamples:")
        print('  python rag_query.py text   "What is the uncanny?"')
        print('  python rag_query.py vlm    "Describe this artwork." artwork.jpg')
        print('  python rag_query.py hybrid "How does this relate to Freud?" artwork.jpg')
        sys.exit(1)

    mode     = sys.argv[1].lower()
    question = sys.argv[2]

    if mode == "text":
        result = query_text_rag(question)

    elif mode == "vlm":
        if len(sys.argv) < 4:
            print("ERROR: vlm mode requires an image path.")
            sys.exit(1)
        result = query_vlm_only(question, sys.argv[3])

    elif mode == "hybrid":
        if len(sys.argv) < 4:
            print("ERROR: hybrid mode requires an image path.")
            sys.exit(1)
        result = query_hybrid(question, sys.argv[3])

    else:
        print(f"Unknown mode: '{mode}'  — use: text | vlm | hybrid")
        sys.exit(1)

    print_result(result)

################################################################
### Image Pipeline
### Artwork images -> LLaVA description -> BGE embed -> PostgreSQL
###
### Usage (CLI):
###   python image_pipeline.py ingest  <image.jpg>
###   python image_pipeline.py search  "painting with dark colours"
###   python image_pipeline.py list
###   python image_pipeline.py delete  <filename>
###
### Usage (as module):
###   from image_pipeline import ingest_image, search_images
################################################################

import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional

# Windows console UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from embed_pipeline import get_embed_model
from db import init_db, insert_artwork, search_artworks, list_artworks, delete_artwork

# ── Config ────────────────────────────────────────────────────
# Folder where uploaded artwork images are stored permanently
ARTWORK_STORE = str(Path(__file__).parent / "artwork_store")

# LLaVA prompt — art-history focused description
ART_DESCRIPTION_PROMPT = (
    "USER: <image>\n"
    "You are an expert art historian. Analyze this artwork and provide a detailed description covering:\n"
    "1. Subject matter and narrative (what is depicted)\n"
    "2. Style and artistic movement (e.g. Baroque, Surrealism, Impressionism)\n"
    "3. Composition and use of space\n"
    "4. Color palette and lighting\n"
    "5. Mood, emotion, and symbolic elements\n"
    "6. Estimated period or artist style if recognizable\n"
    "Be specific and use art history terminology.\n"
    "ASSISTANT:"
)


# ── LLaVA description generator ──────────────────────────────
def describe_image(image_path: str, custom_prompt: Optional[str] = None) -> str:
    """
    Use LLaVA 1.5-7B to generate a rich art-history description
    of the given image.

    Parameters
    ----------
    image_path    : path to the image file
    custom_prompt : override the default art description prompt

    Returns
    -------
    str — LLaVA's description of the artwork
    """
    # Import here to avoid loading GPU model at module import time
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    MODEL_ID = "llava-hf/llava-1.5-7b-hf"

    # Reuse cached model from rag_query if already loaded
    try:
        from rag_query import load_llava
        processor, model = load_llava()
    except Exception:
        # Fallback: load fresh (slower, but safe)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for LLaVA inference.")

        bnb_cfg   = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        model     = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        ).to("cuda")
        model.eval()

    prompt = custom_prompt or ART_DESCRIPTION_PROMPT
    image  = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.05,
        )

    full = processor.decode(output_ids[0], skip_special_tokens=True)

    if "ASSISTANT:" in full:
        return full.split("ASSISTANT:")[-1].strip()
    return full.strip()


# ── Main ingestion function ───────────────────────────────────
def ingest_image(
    image_path: str,
    custom_prompt: Optional[str] = None,
) -> dict:
    """
    Full image ingestion pipeline:
      image → LLaVA description → BGE embedding → PostgreSQL

    The image file is copied into artwork_store/ for permanent storage.
    Re-ingesting the same filename replaces the old record.

    Parameters
    ----------
    image_path    : path to the image file
    custom_prompt : override the default art description prompt

    Returns
    -------
    dict with keys: filename, description, stored_path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    filename = Path(image_path).name
    print(f"\n[Image] -- Ingesting: {filename}")

    # -- Step 1: Copy image to permanent store --------------------
    os.makedirs(ARTWORK_STORE, exist_ok=True)
    stored_path = os.path.join(ARTWORK_STORE, filename)
    shutil.copy2(image_path, stored_path)
    print(f"[Image] Stored at: {stored_path}")

    # -- Step 2: LLaVA generates art description ------------------
    print("[Image] Generating art description with LLaVA ...")
    description = describe_image(stored_path, custom_prompt)
    print(f"[Image] Description ({len(description)} chars):")
    print(f"        {description[:200]}...")

    # -- Step 3: Embed the description ----------------------------
    print("[Image] Embedding description with BGE ...")
    model = get_embed_model()
    embedding = model.encode(
        [description],
        normalize_embeddings=True,
    )[0].tolist()

    # -- Step 4: Store in PostgreSQL ------------------------------
    init_db()
    row_id = insert_artwork(filename, stored_path, description, embedding)
    print(f"[Image] OK  stored in PostgreSQL (id={row_id})")

    return {
        "id"         : row_id,
        "filename"   : filename,
        "stored_path": stored_path,
        "description": description,
    }


# ── Search images by text query ───────────────────────────────
def search_images(query: str, top_k: int = 3) -> List[dict]:
    """
    Find the most relevant artwork images for a text query.
    Embeds the query with BGE and searches PostgreSQL by cosine similarity.

    Example:
      search_images("dark surrealist painting with distorted figures")

    Returns
    -------
    List of dicts: filename, filepath, description, score
    """
    model     = get_embed_model()
    q_vec     = model.encode([query], normalize_embeddings=True)[0].tolist()
    results   = search_artworks(q_vec, top_k=top_k)
    return results


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import textwrap

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python image_pipeline.py ingest <image.jpg>")
        print("  python image_pipeline.py search <query>")
        print("  python image_pipeline.py list")
        print("  python image_pipeline.py delete <filename>")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("ERROR: provide an image path.")
            sys.exit(1)
        result = ingest_image(sys.argv[2])
        print("\n" + "=" * 60)
        print(f"  File        : {result['filename']}")
        print(f"  Stored at   : {result['stored_path']}")
        print(f"  Description :")
        for line in textwrap.wrap(result["description"], width=56):
            print(f"    {line}")
        print("=" * 60)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("ERROR: provide a search query.")
            sys.exit(1)
        query   = sys.argv[2]
        results = search_images(query, top_k=5)
        print(f"\n-- Search results for: '{query}' --------------------")
        if not results:
            print("  No artworks ingested yet.")
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r['filename']}  similarity={round(float(r['score']), 4)}")
            preview = r["description"][:150].replace("\n", " ")
            print(f"      {preview}...")

    elif cmd == "list":
        artworks = list_artworks()
        print(f"\n-- Ingested artworks ({len(artworks)}) -----------------------")
        if not artworks:
            print("  None yet.")
        for a in artworks:
            print(f"  {a['filename']}  |  {str(a['created_at'])[:19]}")

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("ERROR: provide a filename.")
            sys.exit(1)
        n = delete_artwork(sys.argv[2])
        if n:
            print(f"Deleted '{sys.argv[2]}' from database.")
        else:
            print(f"'{sys.argv[2]}' not found.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

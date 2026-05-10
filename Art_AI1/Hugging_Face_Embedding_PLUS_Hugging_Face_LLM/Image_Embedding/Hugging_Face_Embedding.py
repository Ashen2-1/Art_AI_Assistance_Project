import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import chromadb
from PIL import Image
import pytesseract

from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ocr_text_and_flag(img: Image.Image, lang: str = "eng") -> Tuple[str, bool]:
    # 输出每个词的置信度
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    words = []
    confs = []
    for w, c in zip(data["text"], data["conf"]):
        w = (w or "").strip()
        try:
            c = float(c)
        except:
            c = -1.0
        if w:
            words.append(w)
            confs.append(c)

    text = " ".join(words).strip()
    # 规则：有足够长度 + 平均置信度不太低 => 认为“有字”
    avg_conf = (sum([x for x in confs if x >= 0]) / max(1, sum(1 for x in confs if x >= 0))) if confs else -1
    normalized = re.sub(r"\s+", " ", text)

    has_text = (len(normalized) >= 20) and (avg_conf >= 30)  # 你可以调阈值
    return normalized, has_text

CLIP_ID = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

@torch.no_grad()
def clip_image_emb(img: Image.Image) -> List[float]:
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats[0].detach().cpu().tolist()

@torch.no_grad()
def clip_text_emb(text: str) -> List[float]:
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats[0].detach().cpu().tolist()

# -----------------------------
# 3) 图像理解：先用 caption 模型（轻量，3060Ti OK）
#    之后再把 caption + OCR 给你现有的文本LLM（Phi-3-mini）
# -----------------------------
captioner = pipeline(
    task="image-to-text",
    model="Salesforce/blip-image-captioning-large",
    device=0 if DEVICE == "cuda" else -1,
)

#
text_llm = pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device=0 if DEVICE == "cuda" else -1,
    dtype=torch.bfloat16 if DEVICE == "cuda" else None,  # 新版用 dtype，别用 torch_dtype
)

def build_answer(question: str, ocr: str, caption: str) -> str:
    system = (
        "You are an AI assistant for art history research.\n"
        "Answer using ONLY the provided evidence.\n"
        "If the evidence is insufficient, say you don't know.\n"
        "Keep your answer within 3–5 short paragraphs.\n"
    )
    evidence = f"OCR_TEXT:\n{ocr if ocr else '(none)'}\n\nIMAGE_CAPTION:\n{caption if caption else '(none)'}\n"
    prompt = f"{system}\nEVIDENCE:\n{evidence}\nQUESTION:\n{question}\n\nANSWER:\n"

    out = text_llm(
        prompt,
        max_new_tokens=350,
        do_sample=False,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()

# -----------------------------
# 4) Chroma：用 PersistentClient，避免每次重跑都重新embedding
# -----------------------------
CHROMA_DIR = "./chroma_img_db"
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="images_clip")

def ingest_images(img_path_or_dir: str):
    p = Path(img_path_or_dir)

    # 既支持：传单个图片；也支持：传一个目录
    if p.is_file():
        img_paths = [p]
    else:
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            img_paths.extend(p.glob(ext))  # 只扫当前目录
            # 如果你想递归子目录：用 p.rglob(ext)

    if not img_paths:
        raise ValueError(f"No images found in: {img_path_or_dir}")

    ids, embeddings, metadatas, documents = [], [], [], []

    for img_file in img_paths:
        img = Image.open(img_file).convert("RGB")

        ocr, has_text = ocr_text_and_flag(img, lang="eng")   # 这步通常CPU（tesseract）
        emb = clip_image_emb(img)                            # 这步GPU（上面我们已改好）

        ids.append(img_file.name)
        embeddings.append(emb)
        metadatas.append({
            "path": str(img_file),
            "has_text": bool(has_text),
            "ocr": (ocr or "")[:5000],
        })
        documents.append(ocr or "")

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    print(f"Ingested {len(ids)} images into Chroma at {CHROMA_DIR}")

def query_and_answer(question: str, top_k: int = 3) -> str:
    qemb = clip_text_emb(question)
    res = collection.query(
        query_embeddings=[qemb],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    hits = list(zip(res["metadatas"][0], res["distances"][0], res["documents"][0]))
    print(json.dumps(
        [{"path": h[0]["path"], "has_text": h[0]["has_text"], "dist": h[1]} for h in hits],
        indent=2
    ))

    # 这里示范：只用最相关的第1张图来回答（你也可以把多张图 evidence 拼起来）
    meta, dist, doc = hits[0]
    img = Image.open(meta["path"]).convert("RGB")
    caption = captioner(img)[0]["generated_text"]

    ocr = meta.get("ocr", "")  # 或 doc
    return build_answer(question, ocr=ocr, caption=caption)

if __name__ == "__main__":
    IMG_DIR = "1.webp"   # <- 改这里
    ingest_images(IMG_DIR)

    q = "Tell me the style and approximate era of this artwork."
    ans = query_and_answer(q, top_k=3)
    print("\n===== ANSWER =====\n")
    print(ans)
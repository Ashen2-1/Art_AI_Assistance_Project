################################################################
### Phase II - PDF Pipeline
### Extracts text from any PDF and returns clean chunks for embedding.
###
### OCR Modes:
###   "printed"  (default) - Tesseract with image preprocessing
###                          Best for: scanned books, typed docs, photocopies
###   "cursive"            - TrOCR (microsoft/trocr-large-handwritten)
###                          Best for: handwritten notes, cursive letters,
###                                    historical manuscripts
###
### Extraction chain (both modes):
###   1. pdfplumber  -> best for digital PDFs (layout + tables)
###   2. OCR         -> printed: Tesseract | cursive: TrOCR
###   3. PyPDFLoader -> final fallback
################################################################

import os
from typing import List

import numpy as np
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from pypdf import PdfReader

# ── Tesseract binary (Windows) ────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Tuning knobs ──────────────────────────────────────────────
MIN_CHARS_PER_PAGE    = 100   # avg chars/page below this → treat as scanned
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200
OCR_DPI               = 400   # higher DPI = more detail for Tesseract


# =============================================================
# A. PAGE / COLUMN SPLITTER
# =============================================================

def _find_brightest_gap(image: Image.Image, search_start: int, search_end: int) -> int:
    """
    Find the x-coordinate of the brightest (whitest) vertical strip
    in the given horizontal range. Used to locate column gaps and
    book spine gaps.
    """
    gray           = np.array(image.convert("L"))
    col_brightness = gray.mean(axis=0)
    center_region  = col_brightness[search_start:search_end]
    window         = max(1, (search_end - search_start) // 20)
    smoothed       = np.convolve(center_region, np.ones(window) / window, mode="same")
    split_local    = int(np.argmax(smoothed))
    return search_start + split_local, float(smoothed[split_local])


def _split_portrait_columns(image: Image.Image) -> List[Image.Image]:
    """
    Detect and split a portrait page that has TWO text columns
    (e.g. academic papers, journal articles).

    Works by finding the brightest vertical strip in the center
    25-75% of the page width — the white gap between columns.

    Only splits if the gap is significantly brighter than the
    surrounding content (avoids splitting single-column pages).
    """
    width, height = image.size

    search_start = int(width * 0.25)
    search_end   = int(width * 0.75)

    split_point, gap_brightness = _find_brightest_gap(image, search_start, search_end)

    # Global average page brightness for comparison
    gray         = np.array(image.convert("L"))
    page_avg     = float(gray.mean())

    # Only split if the detected gap is noticeably whiter than average
    # (gap brightness > page average + 15 AND > 220 out of 255)
    if gap_brightness > page_avg + 15 and gap_brightness > 220:
        left  = image.crop((0, 0, split_point, height))
        right = image.crop((split_point, 0, width, height))
        print(f"[PDF Pipeline]   Two-column portrait detected -> "
              f"split at x={split_point} (gap brightness={gap_brightness:.1f})")
        return [left, right]

    # Single-column page — return unchanged
    return [image]


def _split_two_page_scan(image: Image.Image) -> List[Image.Image]:
    """
    Detect and split scanned pages that contain more than one column
    of text. Handles two layouts:

    1. LANDSCAPE (aspect >= 1.2) — open book spread:
       Two physical book pages photographed together side by side.
       Splits at the book spine gap in the center 30-70% of width.

    2. PORTRAIT (aspect < 1.2) — two-column article/paper:
       A single physical page with two text columns.
       Splits at the white gutter gap in the center 25-75% of width.
       Only splits when a clear gap is detected (avoids false positives
       on single-column pages).
    """
    width, height = image.size
    aspect        = width / height

    if aspect >= 1.2:
        # ── Landscape: open book spread ──────────────────────
        split_point, _ = _find_brightest_gap(image, int(width * 0.30), int(width * 0.70))
        left_page  = image.crop((0, 0, split_point, height))
        right_page = image.crop((split_point, 0, width, height))
        print(f"[PDF Pipeline]   Book spread detected -> split at x={split_point} "
              f"(left: {left_page.size}, right: {right_page.size})")
        return [left_page, right_page]

    else:
        # ── Portrait: may be two-column layout ───────────────
        return _split_portrait_columns(image)


# =============================================================
# B. IMAGE PREPROCESSING  (improves printed OCR accuracy)
# =============================================================

def _preprocess_for_printed_ocr(image: Image.Image) -> Image.Image:
    """
    Apply a 4-step preprocessing chain that significantly improves
    Tesseract accuracy on scanned / photographed pages:

      1. Grayscale      - remove colour noise
      2. Contrast x2    - make ink vs paper difference sharper
      3. Sharpen        - crisp up blurry letter edges
      4. Binarise       - convert to pure black/white (removes shadows)

    Returns a grayscale PIL Image ready for pytesseract.
    """
    # 1. Grayscale
    image = image.convert("L")

    # 2. Boost contrast (factor 2.0 works well for most scans)
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # 3. Sharpen edges
    image = image.filter(ImageFilter.SHARPEN)

    # 4. Binarise: pixels darker than 140 -> black, rest -> white
    #    Adaptive threshold handles uneven lighting better than a global cutoff
    image = image.point(lambda p: 0 if p < 140 else 255)

    return image


# =============================================================
# C. LINE SEGMENTATION  (required for TrOCR cursive mode)
# =============================================================

def _segment_lines(page_image: Image.Image, min_line_height: int = 12) -> List[Image.Image]:
    """
    Split a full page into individual horizontal line strips using
    a horizontal projection profile (sum of dark pixels per row).

    TrOCR was trained on single-line images, so feeding it one line
    at a time gives much better results than the full page.

    Parameters
    ----------
    page_image      : full-page PIL Image
    min_line_height : ignore strips shorter than this (noise filter)

    Returns
    -------
    List of PIL Images, one per text line.
    Falls back to the full page if segmentation finds nothing.
    """
    gray   = np.array(page_image.convert("L"))
    binary = (gray < 140).astype(np.uint8)          # 1 where ink is

    # How many dark pixels in each row?
    row_sum    = binary.sum(axis=1)
    threshold  = binary.shape[1] * 0.005            # >0.5% dark = text row
    is_text    = row_sum > threshold

    line_images = []
    in_line     = False
    start       = 0

    for row_idx, has_text in enumerate(is_text):
        if has_text and not in_line:
            start   = row_idx
            in_line = True
        elif not has_text and in_line:
            in_line = False
            height  = row_idx - start
            if height >= min_line_height:
                pad = 4                              # small vertical padding
                y1  = max(0, start - pad)
                y2  = min(page_image.height, row_idx + pad)
                line_images.append(page_image.crop((0, y1, page_image.width, y2)))

    # Catch last line if page ends while still in a text block
    if in_line and (page_image.height - start) >= min_line_height:
        y1 = max(0, start - 4)
        line_images.append(page_image.crop((0, y1, page_image.width, page_image.height)))

    return line_images if line_images else [page_image]   # fallback: full page


# =============================================================
# D. OCR ENGINES
# =============================================================

def _extract_ocr_printed(pdf_path: str) -> str:
    """
    PRINTED / SCANNED PDF OCR
    --------------------------
    Pipeline:
      pdf2image (400 DPI)
        -> split two-page book spreads into left + right pages
        -> preprocess each page (grayscale, contrast, sharpen, binarise)
        -> Tesseract (LSTM engine, single-column block mode)

    Splitting two-page scans is the most important fix for book PDFs:
    without it Tesseract reads across both columns and mixes them up.

    Tesseract config flags:
      --oem 3   : use both legacy + LSTM engines (best accuracy)
      --psm 6   : assume a uniform block of text per page
    """
    print("[PDF Pipeline] Converting pages to images (400 DPI) ...")
    images      = convert_from_path(pdf_path, dpi=OCR_DPI)
    tess_config = r"--oem 3 --psm 6"
    full_text   = ""
    page_num    = 1

    for i, image in enumerate(images):
        print(f"[PDF Pipeline]   Scan {i + 1}/{len(images)} ...")

        # Split open-book spreads into individual pages
        sub_pages = _split_two_page_scan(image)

        for sub_page in sub_pages:
            processed  = _preprocess_for_printed_ocr(sub_page)
            page_text  = pytesseract.image_to_string(
                processed, lang="eng", config=tess_config
            )
            full_text += f"[Page {page_num}]\n{page_text}\n\n"
            page_num  += 1

    return full_text


def _extract_ocr_cursive(pdf_path: str) -> str:
    """
    CURSIVE / HANDWRITTEN PDF OCR
    ──────────────────────────────
    Uses Microsoft TrOCR (Transformer OCR) - a Vision Encoder-Decoder
    model fine-tuned on handwritten text. Far superior to Tesseract
    for cursive, historical manuscripts, and personal letters.

    Pipeline:
      pdf2image (300 DPI) -> segment into lines -> TrOCR each line -> join

    Model: microsoft/trocr-large-handwritten
    Device: CUDA (RTX 5070) for fast inference
    """
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"[PDF Pipeline] Loading TrOCR handwriting model on {device} ...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model     = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("[PDF Pipeline] TrOCR model ready.")

    images    = convert_from_path(pdf_path, dpi=300)
    full_text = ""

    for i, page_img in enumerate(images):
        print(f"[PDF Pipeline]   Cursive OCR page {i + 1}/{len(images)} ...")
        lines     = _segment_lines(page_img)
        page_text = ""

        for line_img in lines:
            # TrOCR expects RGB input
            pixel_values = processor(
                line_img.convert("RGB"), return_tensors="pt"
            ).pixel_values.to(device, dtype=dtype)

            with torch.inference_mode():
                generated_ids = model.generate(pixel_values, max_new_tokens=128)

            line_text  = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            page_text += line_text + "\n"

        full_text += f"[Page {i + 1}]\n{page_text}\n\n"

    return full_text


# =============================================================
# E. DIGITAL PDF EXTRACTION  (pdfplumber)
# =============================================================

def _extract_page_text_smart(page) -> str:
    """
    Column-aware text extraction for a single pdfplumber page.

    Problem: pdfplumber's default extract_text() reads words in
    Y-order across the full page width. For two-column layouts this
    mixes both columns line by line, producing garbled output.

    Fix:
      1. Use extract_words() to get bounding boxes of every word.
      2. Count words in the left half vs right half.
      3. If both halves have significant content (>25% each) the page
         is two-column: crop left and right separately and extract
         each column in its own Y-sorted order.
      4. Otherwise fall back to default extract_text().
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return page.extract_text() or ""

    x_mid       = page.width / 2
    total       = len(words)
    left_count  = sum(1 for w in words if float(w["x0"]) < x_mid)
    right_count = sum(1 for w in words if float(w["x0"]) >= x_mid)

    # Two-column heuristic: both sides must have at least 25% of words
    is_two_col = (
        total > 10
        and (left_count  / total) > 0.25
        and (right_count / total) > 0.25
    )

    if is_two_col:
        # Crop each column and extract text independently
        left_col  = page.crop((0,     0, x_mid,       page.height))
        right_col = page.crop((x_mid, 0, page.width,  page.height))
        left_text  = left_col.extract_text()  or ""
        right_text = right_col.extract_text() or ""
        return (left_text + "\n\n" + right_text).strip()

    return page.extract_text() or ""


def _extract_pdfplumber(pdf_path: str) -> tuple:
    """
    Column-aware pdfplumber extraction.
    Handles single-column and two-column digital PDFs correctly.
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = _extract_page_text_smart(page)
            if text:
                full_text += f"[Page {i + 1}]\n{text}\n\n"
    return full_text, num_pages


def _extract_pypdf(pdf_path: str) -> str:
    """Final fallback: pypdf PdfReader (no langchain needed)."""
    reader = PdfReader(pdf_path)
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _is_scanned(text: str, num_pages: int) -> bool:
    """True if avg chars/page < MIN_CHARS_PER_PAGE (image-only PDF)."""
    if num_pages == 0:
        return True
    return (len(text.strip()) / num_pages) < MIN_CHARS_PER_PAGE


# =============================================================
# E. MAIN EXTRACTION ENTRY POINT
# =============================================================

def extract_pdf_text(pdf_path: str, ocr_mode: str = "printed") -> str:
    """
    Full extraction chain with selectable OCR engine.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    ocr_mode : str
        "printed"  (default) - Tesseract + preprocessing
        "cursive"            - TrOCR handwriting model

    Returns
    -------
    str  - all text from every page, with [Page N] headers.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    mode_label = "Cursive/Handwriting (TrOCR)" if ocr_mode == "cursive" else "Printed/Scanned (Tesseract)"
    print(f"\n[PDF Pipeline] -- Processing : {os.path.basename(pdf_path)}")
    print(f"[PDF Pipeline]    OCR mode   : {mode_label}")

    # -- Step 1: pdfplumber (always tried first) ----------------------
    try:
        text, num_pages = _extract_pdfplumber(pdf_path)
        if not _is_scanned(text, num_pages):
            avg = len(text.strip()) // max(num_pages, 1)
            print(f"[PDF Pipeline] OK pdfplumber  |  {num_pages} pages  |  ~{avg} chars/page")
            return text
        else:
            avg = len(text.strip()) // max(num_pages, 1)
            print(f"[PDF Pipeline] Scanned PDF detected (~{avg} chars/page) -> switching to OCR")
    except Exception as e:
        print(f"[PDF Pipeline] pdfplumber error: {e}  -> trying OCR")

    # -- Step 2: OCR (engine chosen by ocr_mode) ---------------------
    try:
        if ocr_mode == "cursive":
            text = _extract_ocr_cursive(pdf_path)
        else:
            text = _extract_ocr_printed(pdf_path)
        print(f"[PDF Pipeline] OK OCR complete  |  {len(text)} chars extracted")
        return text
    except Exception as e:
        print(f"[PDF Pipeline] OCR error: {e}  -> falling back to PyPDFLoader")

    # -- Step 3: PyPDFLoader fallback --------------------------------
    text = _extract_pypdf(pdf_path)
    print(f"[PDF Pipeline] OK PyPDFLoader fallback  |  {len(text)} chars extracted")
    return text


# =============================================================
# F. CHUNKER
# =============================================================

def chunk_text(
    text: str,
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split extracted text into overlapping chunks for embedding.

    Returns
    -------
    List[str]  - chunks ready to pass to the embedding model.
    """
    # Simple recursive character splitter (no langchain dependency)
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)
    print(f"[PDF Pipeline] OK {len(chunks)} chunks  (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list,
) -> List[str]:
    """
    Recursively split text using a list of separators (largest -> smallest).
    Mimics LangChain RecursiveCharacterTextSplitter behaviour.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep) if sep else list(text)

    chunks: List[str] = []
    current = ""
    for part in parts:
        piece = (current + sep + part) if current else part
        if len(piece) <= chunk_size:
            current = piece
        else:
            if current.strip():
                if len(current) > chunk_size and remaining:
                    chunks.extend(_recursive_split(current, chunk_size, chunk_overlap, remaining))
                else:
                    chunks.append(current)
            current = part

    if current.strip():
        if len(current) > chunk_size and remaining:
            chunks.extend(_recursive_split(current, chunk_size, chunk_overlap, remaining))
        else:
            chunks.append(current)

    # Apply overlap: each chunk starts chunk_overlap chars before the next chunk begins
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-chunk_overlap:]
            overlapped.append(tail + chunks[i])
        return [c for c in overlapped if c.strip()]

    return [c for c in chunks if c.strip()]


# =============================================================
# G. PUBLIC API
# =============================================================

def process_pdf(
    pdf_path: str,
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ocr_mode: str      = "printed",
) -> List[str]:
    """
    Phase II PDF Pipeline - one call, full pipeline.

    Parameters
    ----------
    pdf_path      : path to the PDF file
    chunk_size    : characters per chunk   (default 1000)
    chunk_overlap : overlap between chunks (default 200)
    ocr_mode      : "printed"  -> Tesseract + preprocessing  (default)
                    "cursive"  -> TrOCR handwriting model

    Returns
    -------
    List[str]  - text chunks ready for embedding.

    Examples
    --------
    # Scanned book or typed document:
    chunks = process_pdf("catalogue.pdf")

    # Handwritten letter or cursive manuscript:
    chunks = process_pdf("letter.pdf", ocr_mode="cursive")
    """
    text   = extract_pdf_text(pdf_path, ocr_mode=ocr_mode)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    return chunks


# =============================================================
# Quick test:
#   python pdf_pipeline.py <file.pdf>           <- printed mode
#   python pdf_pipeline.py <file.pdf> cursive   <- cursive mode
# =============================================================

if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    target   = sys.argv[1] if len(sys.argv) > 1 else "Sigmund.pdf"
    mode     = sys.argv[2] if len(sys.argv) > 2 else "printed"

    chunks   = process_pdf(target, ocr_mode=mode)

    print("\n" + "=" * 60)
    print(f"  RESULT  : {len(chunks)} chunks")
    print(f"  FILE    : {os.path.basename(target)}")
    print(f"  MODE    : {mode}")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} / {len(chunks)} ---")
        print(chunk)

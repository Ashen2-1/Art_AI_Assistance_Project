"""
Images for the Photo Detective: photographs, paintings, drawings, prints.

- prepare_image(): any common format -> RGB JPEG, longest side <= 2048 px
  (Gemini accepts JPEG/PNG/WEBP; TIFF/BMP/GIF are converted).
- describe_for_index(): a detailed, searchable description + transcription
  of visible text, so images can be found by search and compared with
  other images and documents in the same workspace.
"""

import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from gemini_client import generate_multimodal

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
MAX_SIDE = 2048

INDEX_PROMPT = """
You are cataloguing an image for a historical research archive. It may be a
photograph (daguerreotype, calotype, albumen print, stereograph, modern
photo), a painting, a drawing or sketch, a print or engraving.

Write a precise, factual catalogue description in plain text with these
headings, one per line followed by the content:

Medium and technique:
Subject:
People (for each: position, apparent age, sex, clothing, hair/facial hair, pose, distinctive features):
Objects and setting:
Visible text (transcribe exactly, including inscriptions, captions, signatures, stamps, numbers; say "none" if none):
Clues to date:
Clues to place:
Clues to maker (photographer/artist, studio marks, style):
Condition:
Distinctive features useful for matching with other images:

Describe only what is visible. Do not guess names unless they are written
in the image. Be specific (e.g. "high stand-up collar, double-breasted
frock coat", not "old clothes").
""".strip()


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def prepare_image(path: str, out_dir: str, max_side: int = MAX_SIDE) -> str:
    """Normalise to an RGB JPEG that the vision model accepts."""
    with Image.open(path) as image:
        # JPEG: decode straight at a reduced size. A 48 MP phone photo is
        # ~140 MB once decoded; draft() keeps it near the size we need.
        if image.format == "JPEG":
            image.draft("RGB", (max_side, max_side))
        image = ImageOps.exif_transpose(image)
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        target = Path(out_dir) / f"{Path(path).stem}-prepared.jpg"
        image.save(target, "JPEG", quality=90)
    return str(target)


def crop_region(path: str, box: dict, out_dir: str, pad: float = 0.12) -> str:
    """
    Crop the region the researcher marked. box = {x, y, w, h} as fractions
    (0-1) of the image. A little padding keeps hair, collars and hands.
    """
    with Image.open(path) as image:
        # Decode big JPEGs at a reduced scale (>= 2560 px: plenty for a face crop, a
        # fraction of the memory), and apply the EXIF rotation so the box
        # matches the picture the researcher saw in the browser.
        if image.format == "JPEG":
            image.draft("RGB", (2560, 2560))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        width, height = image.size
        x = max(0.0, min(1.0, float(box.get("x", 0))))
        y = max(0.0, min(1.0, float(box.get("y", 0))))
        w = max(0.01, min(1.0 - x, float(box.get("w", 1))))
        h = max(0.01, min(1.0 - y, float(box.get("h", 1))))
        left = max(0, int((x - w * pad) * width))
        top = max(0, int((y - h * pad) * height))
        right = min(width, int((x + w * (1 + pad)) * width))
        bottom = min(height, int((y + h * (1 + pad)) * height))
        region = image.crop((left, top, max(right, left + 8), max(bottom, top + 8)))
        # Small faces benefit from upscaling before the vision model sees them.
        if max(region.size) < 512:
            factor = 512 / max(region.size)
            region = region.resize((int(region.width * factor), int(region.height * factor)), Image.LANCZOS)
        region.thumbnail((MAX_SIDE, MAX_SIDE))
        target = Path(out_dir) / f"{Path(path).stem}-focus.jpg"
        region.save(target, "JPEG", quality=92)
    return str(target)


def describe_for_index(path: str) -> str:
    """Searchable description of one image (used when the image is indexed)."""
    with tempfile.TemporaryDirectory(prefix="nexo_img_") as temp_dir:
        prepared = prepare_image(path, temp_dir)
        return generate_multimodal(
            INDEX_PROMPT,
            [prepared],
            temperature=0.1,
            max_output_tokens=2048,
        ).strip()

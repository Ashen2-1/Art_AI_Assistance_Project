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


def prepare_image(path: str, out_dir: str) -> str:
    """Normalise to an RGB JPEG that the vision model accepts."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        target = Path(out_dir) / f"{Path(path).stem}-prepared.jpg"
        image.save(target, "JPEG", quality=90)
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

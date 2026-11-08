################################################################
### Document Pipeline
### Any supported file -> readable, location-labelled chunks
###
### Every chunk starts with a location marker on its first line,
### the same convention the PDF pipeline already uses:
###
###   [Page 4]                    PDF
###   [Slide 3]                   PowerPoint (.pptx)
###   [Section 2: Methods]        Word (.docx)
###   [Sheet Budget, rows 1-40]   Excel (.xlsx) / CSV
###   [Lines 120-180]             text, Markdown, code
###   [Cell 5]                    Jupyter notebook
###
### The API turns this marker into structured location data so
### the frontend can jump to the right page / slide / lines and
### highlight the cited passage.
################################################################

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

DEFAULT_CHUNK_SIZE = 1000

PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx"}
POWERPOINT_EXTENSIONS = {".pptx"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
CSV_EXTENSIONS = {".csv", ".tsv"}
NOTEBOOK_EXTENSIONS = {".ipynb"}

# Legacy Office formats need LibreOffice to convert first.
LEGACY_CONVERSIONS = {
    ".doc": "docx",
    ".ppt": "pptx",
    ".xls": "xlsx",
    ".rtf": "docx",
    ".odt": "docx",
    ".odp": "pptx",
    ".ods": "xlsx",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".json", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".tex", ".bib",
    ".html", ".htm", ".css", ".scss",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".scala", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".m", ".r",
    ".sql", ".sh", ".bash", ".ps1", ".bat", ".lua", ".dart",
    ".vue", ".svelte",
}

SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS
    | WORD_EXTENSIONS
    | POWERPOINT_EXTENSIONS
    | EXCEL_EXTENSIONS
    | CSV_EXTENSIONS
    | NOTEBOOK_EXTENSIONS
    | TEXT_EXTENSIONS
    | set(LEGACY_CONVERSIONS)
)

MAX_SHEET_ROWS = 5000      # per sheet, keeps huge workbooks bounded
MAX_TEXT_CHARS = 2_000_000  # ~2 MB of text


class UnsupportedDocumentError(ValueError):
    """Raised for file types that cannot be indexed."""


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# ------------------------------------------------------------------
# Shared chunking helpers
# ------------------------------------------------------------------

def _split_long_text(text: str, chunk_size: int) -> List[str]:
    """Split one long paragraph on sentence boundaries, then hard-wrap."""

    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        while len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, chunk_size)
            cut = cut if cut > chunk_size // 2 else chunk_size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return [piece for piece in pieces if piece.strip()]


def chunk_sections(
    sections: Iterable[Tuple[str, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[str]:
    """
    sections: (location_label, text) pairs, e.g. ("Slide 3", "...").
    Paragraphs inside one section are packed together up to
    chunk_size; a chunk never mixes two locations, so every chunk
    has exactly one, accurate location marker.
    """

    chunks: List[str] = []

    for label, text in sections:
        marker = f"[{label}]"
        budget = max(200, chunk_size - len(marker) - 1)

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text or "")
            if paragraph.strip()
        ]

        current = ""

        for paragraph in paragraphs:
            parts = (
                _split_long_text(paragraph, budget)
                if len(paragraph) > budget
                else [paragraph]
            )

            for part in parts:
                candidate = f"{current}\n\n{part}" if current else part

                if len(candidate) <= budget:
                    current = candidate
                else:
                    if current:
                        chunks.append(f"{marker}\n{current}")
                    current = part

        if current:
            chunks.append(f"{marker}\n{current}")

    return chunks


def chunk_lines(
    lines: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    label_prefix: str = "Lines",
) -> List[str]:
    """
    Pack consecutive lines (code, logs, plain text) into chunks and
    label each with its real 1-based line range, e.g. [Lines 40-72].
    """

    chunks: List[str] = []
    start = 0
    buffer: List[str] = []
    size = 0

    def flush(end_index: int):
        if any(line.strip() for line in buffer):
            body = "\n".join(buffer).strip("\n")
            chunks.append(f"[{label_prefix} {start + 1}-{end_index}]\n{body}")

    for index, line in enumerate(lines):
        line_length = len(line) + 1

        if buffer and size + line_length > chunk_size:
            flush(index)
            buffer, size, start = [], 0, index

        # A single very long line (minified code) is hard-wrapped.
        if line_length > chunk_size:
            for offset in range(0, len(line), chunk_size):
                piece = line[offset:offset + chunk_size]
                chunks.append(f"[{label_prefix} {index + 1}-{index + 1}]\n{piece}")
            start = index + 1
            continue

        buffer.append(line)
        size += line_length

    if buffer:
        flush(len(lines))

    return chunks


# ------------------------------------------------------------------
# Format readers
# ------------------------------------------------------------------

def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()[:MAX_TEXT_CHARS * 4]

    for encoding in ("utf-8-sig", "utf-16", "gb18030", "latin-1"):
        try:
            text = raw.decode(encoding)
            if encoding == "utf-16" and "\x00" in text[:200]:
                continue
            return text[:MAX_TEXT_CHARS]
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]


def _docx_sections(path: Path) -> List[Tuple[str, str]]:
    """
    Word has no fixed pages, so we use headings as the location:
    each heading starts a new numbered section. Tables are kept
    as pipe-separated rows inside their section.
    """

    from docx import Document  # python-docx

    document = Document(str(path))
    body = document.element.body

    sections: List[Tuple[str, str]] = []
    section_number = 1
    section_title: Optional[str] = None
    buffer: List[str] = []

    def label() -> str:
        title = (section_title or "").strip()
        if title:
            title = title[:60]
            return f"Section {section_number}: {title}"
        return f"Section {section_number}"

    def flush():
        text = "\n\n".join(part for part in buffer if part.strip())
        if text.strip():
            sections.append((label(), text))

    paragraph_lookup = {p._p: p for p in document.paragraphs}
    table_lookup = {t._tbl: t for t in document.tables}

    for element in body.iterchildren():
        if element in paragraph_lookup:
            paragraph = paragraph_lookup[element]
            text = paragraph.text.strip()

            if not text:
                continue

            style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
            is_heading = style.startswith("heading") or style == "title"

            if is_heading:
                if any(part.strip() for part in buffer):
                    flush()
                    section_number += 1
                buffer = [text]
                section_title = text
            else:
                buffer.append(text)

        elif element in table_lookup:
            table = table_lookup[element]
            rows = []

            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # Merged cells repeat; drop consecutive duplicates.
                deduped = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
                if any(deduped):
                    rows.append(" | ".join(deduped))

            if rows:
                buffer.append("\n".join(rows))

    flush()
    return sections


def _pptx_sections(path: Path) -> List[Tuple[str, str]]:
    from pptx import Presentation  # python-pptx

    presentation = Presentation(str(path))
    sections: List[Tuple[str, str]] = []

    for number, slide in enumerate(presentation.slides, start=1):
        parts: List[str] = []

        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                text = "\n".join(
                    paragraph_text
                    for paragraph_text in (
                        "".join(run.text for run in paragraph.runs).strip()
                        for paragraph in shape.text_frame.paragraphs
                    )
                    if paragraph_text
                )
                if text:
                    parts.append(text)

            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [
                    " | ".join(cell.text.strip() for cell in row.cells)
                    for row in shape.table.rows
                ]
                parts.append("\n".join(row for row in rows if row.strip(" |")))

        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"Speaker notes: {notes}")

        text = "\n\n".join(part for part in parts if part.strip())

        if text:
            sections.append((f"Slide {number}", text))

    return sections


def _rows_to_chunks(
    sheet_name: str,
    rows: List[List[str]],
    chunk_size: int,
) -> List[str]:
    """
    Tables are chunked by row ranges. The header row is repeated in
    every chunk so each chunk can be understood on its own.
    Row numbers are the real spreadsheet row numbers (1-based), even
    when empty rows are skipped.
    """

    numbered = []

    for row_number, row in enumerate(rows, start=1):
        cells = [str(value).strip() if value is not None else "" for value in row]
        if any(cells):
            numbered.append((row_number, cells))

    if not numbered:
        return []

    header_row, header_cells = numbered[0]
    header = " | ".join(header_cells)
    chunks: List[str] = []
    buffer: List[str] = []
    start_row = None
    last_row = header_row
    size = len(header)

    def flush():
        if buffer:
            body = "\n".join([header, *buffer])
            chunks.append(f"[Sheet {sheet_name}, rows {start_row}-{last_row}]\n{body}")

    for row_number, cells in numbered[1:]:
        line = " | ".join(cells)

        if buffer and size + len(line) + 1 > chunk_size:
            flush()
            buffer, size, start_row = [], len(header), None

        if start_row is None:
            start_row = row_number

        buffer.append(line[:chunk_size])
        last_row = row_number
        size += len(line) + 1

    flush()

    if not chunks:
        # A sheet with only a header row: still index it.
        chunks.append(f"[Sheet {sheet_name}, rows {header_row}-{header_row}]\n{header}")

    return chunks


def _xlsx_chunks(path: Path, chunk_size: int) -> List[str]:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    chunks: List[str] = []

    try:
        for sheet in workbook.worksheets:
            rows = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index >= MAX_SHEET_ROWS:
                    break
                rows.append(list(row))
            chunks.extend(_rows_to_chunks(sheet.title, rows, chunk_size))
    finally:
        workbook.close()

    return chunks


def _csv_chunks(path: Path, chunk_size: int) -> List[str]:
    text = _read_text_file(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        pass

    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))[:MAX_SHEET_ROWS]
    return _rows_to_chunks(path.stem, rows, chunk_size)


def _notebook_chunks(path: Path, chunk_size: int) -> List[str]:
    notebook = json.loads(_read_text_file(path))
    sections = []

    for number, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)

        if text.strip():
            kind = cell.get("cell_type", "cell")
            sections.append((f"Cell {number} ({kind})", text))

    return chunk_sections(sections, chunk_size)


def _convert_legacy_office(path: Path, target_ext: str, out_dir: Path) -> Path:
    """
    .doc / .ppt / .xls -> .docx / .pptx / .xlsx with LibreOffice.
    LibreOffice is optional (it is large); without it we explain
    how to save the file in the modern format instead.
    """

    soffice = shutil.which("soffice") or shutil.which("libreoffice")

    if not soffice:
        raise UnsupportedDocumentError(
            f"{path.suffix} files need to be converted first. "
            f"Please save the file as .{target_ext} (File > Save As) and upload it again."
        )

    subprocess.run(
        [soffice, "--headless", "--convert-to", target_ext, "--outdir", str(out_dir), str(path)],
        check=True,
        timeout=180,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    converted = out_dir / f"{path.stem}.{target_ext}"

    if not converted.exists():
        raise UnsupportedDocumentError(f"Could not convert {path.name} to .{target_ext}.")

    return converted


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def process_document(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    ocr_mode: str = "printed",
) -> List[str]:
    """
    Extract and chunk any supported document.
    Returns chunks whose first line is a location marker.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"{extension or 'This'} files are not supported for AI reading yet."
        )

    if extension in PDF_EXTENSIONS:
        from pdf_pipeline import process_pdf

        return process_pdf(str(path), chunk_size=chunk_size, ocr_mode=ocr_mode)

    if extension in LEGACY_CONVERSIONS:
        with tempfile.TemporaryDirectory(prefix="nexo_convert_") as temp_dir:
            converted = _convert_legacy_office(path, LEGACY_CONVERSIONS[extension], Path(temp_dir))
            return process_document(str(converted), chunk_size=chunk_size, ocr_mode=ocr_mode)

    if extension in WORD_EXTENSIONS:
        chunks = chunk_sections(_docx_sections(path), chunk_size)
    elif extension in POWERPOINT_EXTENSIONS:
        chunks = chunk_sections(_pptx_sections(path), chunk_size)
    elif extension in EXCEL_EXTENSIONS:
        chunks = _xlsx_chunks(path, chunk_size)
    elif extension in CSV_EXTENSIONS:
        chunks = _csv_chunks(path, chunk_size)
    elif extension in NOTEBOOK_EXTENSIONS:
        chunks = _notebook_chunks(path, chunk_size)
    else:
        text = _read_text_file(path).replace("\r\n", "\n").replace("\r", "\n")
        chunks = chunk_lines(text.split("\n"), chunk_size)

    chunks = [chunk for chunk in chunks if chunk.split("\n", 1)[-1].strip()]

    if not chunks:
        raise ValueError(f"No readable text was found in {path.name}.")

    print(f"[Document Pipeline] {path.name}: {len(chunks)} chunks")
    return chunks


# ------------------------------------------------------------------
# Location markers -> structured data (used by the API)
# ------------------------------------------------------------------

_LOCATION_PATTERNS = [
    (re.compile(r"^\[Page\s+(\d+)\]"), "page"),
    (re.compile(r"^\[Slide\s+(\d+)\]"), "slide"),
    (re.compile(r"^\[Section\s+(\d+)(?::\s*(.*?))?\]"), "section"),
    (re.compile(r"^\[Sheet\s+(.+?),\s*rows\s+(\d+)-(\d+)\]"), "sheet"),
    (re.compile(r"^\[Lines\s+(\d+)-(\d+)\]"), "lines"),
    (re.compile(r"^\[Cell\s+(\d+)(?:\s*\((.*?)\))?\]"), "cell"),
]


def parse_location(chunk_text: str) -> dict:
    """
    '[Page 4]\\n...' -> {"type": "page", "page": 4, "label": "Page 4"}
    Unknown or missing marker -> {"type": "unknown", "label": ""}
    """

    first_line = (chunk_text or "").lstrip().split("\n", 1)[0].strip()

    for pattern, kind in _LOCATION_PATTERNS:
        match = pattern.match(first_line)

        if not match:
            continue

        label = first_line.strip("[]")

        if kind == "page":
            return {"type": "page", "page": int(match.group(1)), "label": label}
        if kind == "slide":
            return {"type": "slide", "slide": int(match.group(1)), "label": label}
        if kind == "section":
            return {
                "type": "section",
                "section": int(match.group(1)),
                "title": match.group(2) or "",
                "label": label,
            }
        if kind == "sheet":
            return {
                "type": "sheet",
                "sheet": match.group(1),
                "row_start": int(match.group(2)),
                "row_end": int(match.group(3)),
                "label": label,
            }
        if kind == "lines":
            return {
                "type": "lines",
                "line_start": int(match.group(1)),
                "line_end": int(match.group(2)),
                "label": label,
            }
        if kind == "cell":
            return {"type": "cell", "cell": int(match.group(1)), "label": label}

    return {"type": "unknown", "label": ""}


def strip_location_marker(chunk_text: str) -> str:
    text = (chunk_text or "").lstrip()
    first_line, _, rest = text.partition("\n")

    if any(pattern.match(first_line.strip()) for pattern, _ in _LOCATION_PATTERNS):
        return rest.strip()

    return text.strip()

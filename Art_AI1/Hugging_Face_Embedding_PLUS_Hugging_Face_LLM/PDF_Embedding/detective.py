"""
Photo Detective.

Three steps for one image (identify) or two images (compare):

  1. LOOK     a vision model writes structured observations (medium,
              people, clothing, setting, visible text, clues to date /
              place / maker) plus search queries.
  2. SEARCH   those queries are run against every indexed source in the
              workspace: books, PDFs, Word files, spreadsheets and the
              descriptions of other images.
  3. REASON   the answer model weighs observations against the retrieved
              evidence and returns candidates with likelihoods, every
              claim cited as [S1], [S2]...

The answer model follows the user's settings (NEXO / Gemini / OpenAI /
Claude); the vision step uses Gemini (the user's key when connected).
"""

import json
import re
from typing import List, Optional

from gemini_client import generate_multimodal, generate_text
from rag_query import build_context, retrieve

MAX_EVIDENCE = 14

LOOK_SYSTEM = """
You are a meticulous visual analyst for a historical image archive
(photographs, paintings, drawings, prints). You describe what is visible,
precisely and without inventing names. Return only valid JSON.
""".strip()

LOOK_PROMPT_ONE = """
Examine the image and return JSON with exactly these keys:
{
  "medium": "photograph | oil painting | watercolour | drawing | sketch | print/engraving | other",
  "process_or_technique": "e.g. albumen print, calotype, stereograph card, oil on canvas, pencil",
  "subject_summary": "one or two sentences",
  "people": [{"position": "", "apparent_age": "", "sex": "", "clothing": "", "hair_and_face": "", "pose_and_action": "", "distinctive_features": ""}],
  "objects_and_setting": "",
  "visible_text": [{"text": "exact transcription", "where": "e.g. bottom right, on the mount"}],
  "date_clues": ["specific clues with what they suggest"],
  "place_clues": ["..."],
  "maker_clues": ["studio stamps, style, format, signature..."],
  "condition": "",
  "distinctive_features": ["features useful to match this image with others"],
  "search_queries": ["4-6 short queries to search the archive: names or words from visible text, distinctive objects, place and era terms, maker names if written"]
}
{question_line}
""".strip()

LOOK_PROMPT_TWO = """
You are given TWO images, image A (first) and image B (second). Return JSON
with exactly these keys:
{
  "image_a": { same keys as below },
  "image_b": { same keys as below },
  "comparison": {
    "same_person_likelihood": 0-100 or null if no people,
    "same_place_likelihood": 0-100 or null,
    "same_maker_likelihood": 0-100 or null,
    "shared_features": ["specific matching details"],
    "differences": ["specific differences or contradictions"],
    "notes": ""
  },
  "search_queries": ["4-6 short archive search queries covering both images"]
}
Keys for each image: medium, process_or_technique, subject_summary, people
(position, apparent_age, sex, clothing, hair_and_face, pose_and_action,
distinctive_features), objects_and_setting, visible_text (text, where),
date_clues, place_clues, maker_clues, condition, distinctive_features.
Be cautious: likelihoods above 80 need several independent matching details.
{question_line}
""".strip()

REASON_SYSTEM = """
You are a careful art and photo historian acting as a "detective". You
weigh visual observations against documentary evidence. Rules:
- Cite evidence with the labels you are given, written separately: [S1] [S3].
- Never invent sources, names, dates or quotations. If the evidence does
  not name anyone, say so and list what would be needed.
- Likelihoods are calibrated percentages (0-100). Several candidates may
  share the probability; leave room for "someone not in the sources".
- Clearly separate what is SEEN in the image from what is INFERRED.
Return only valid JSON.
""".strip()

REASON_PROMPT = """
Task: {task}
{question_line}

VISUAL OBSERVATIONS (from the image{plural}):
{observations}

EVIDENCE FROM THE WORKSPACE SOURCES:
{evidence}

Return JSON with exactly these keys:
{{
  "summary": "3-5 sentences: what this is and the most likely identification, with citations",
  "candidates": [
    {{"name": "person / place / maker", "kind": "sitter | maker | place | object | other", "likelihood": 0-100,
      "evidence_for": ["reason with citation [S#]"], "evidence_against": ["reason"]}}
  ],
  "unidentified": "if no candidate is supported: what the image shares with the sources and what would identify it, else empty",
  "date_estimate": {{"range": "", "confidence": "low | medium | high", "reasoning": "with citations"}},
  "place_estimate": {{"value": "", "confidence": "low | medium | high", "reasoning": ""}},
  "maker_estimate": {{"value": "", "confidence": "low | medium | high", "reasoning": ""}},
  "similar_items": [{{"source": "file name", "citation": "S#", "shared_features": "what matches"}}],
  "comparison": {compare_shape},
  "transcribed_text": ["exact text visible in the image{plural}"],
  "open_questions": ["what is still uncertain"],
  "next_steps": ["concrete checks: archives, catalogues, comparisons"]
}}
""".strip()


def _parse_json(text: str) -> Optional[dict]:
    """Models sometimes wrap JSON in prose or code fences."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(cleaned[start:end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


def _clamp(value):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def _gather_evidence(queries: List[str], user_id: str, canvas_id: str, source_filters: List[str]) -> List[dict]:
    """Run several searches and keep the best distinct passages."""
    seen = {}
    for query in queries:
        query = str(query or "").strip()
        if not query:
            continue
        try:
            rows = retrieve(
                question=query,
                user_id=user_id,
                canvas_id=canvas_id,
                top_k=4,
                source_filters=source_filters or None,
            )
        except Exception as error:  # one failed query should not stop the analysis
            print(f"[Detective] search failed for {query!r}: {error}")
            continue
        for row in rows:
            key = (row["source"], row["chunk_idx"])
            if key not in seen or row["score"] > seen[key]["score"]:
                seen[key] = row
    best = sorted(seen.values(), key=lambda row: row["score"], reverse=True)[:MAX_EVIDENCE]
    return best


def analyze(
    image_paths: List[str],
    image_names: List[str],
    user_id: str,
    canvas_id: str = "default",
    question: str = "",
    source_filters: Optional[List[str]] = None,
) -> dict:
    compare = len(image_paths) == 2
    question_line = f"The researcher asks: {question.strip()}" if question.strip() else ""

    # 1. LOOK
    look_prompt = (LOOK_PROMPT_TWO if compare else LOOK_PROMPT_ONE).replace("{question_line}", question_line)
    look_text = generate_multimodal(look_prompt, image_paths, system_instruction=LOOK_SYSTEM, temperature=0.1, max_output_tokens=4096, json_output=True)
    observations = _parse_json(look_text) or {"subject_summary": look_text.strip()}

    # 2. SEARCH: the model's queries, the researcher's question, and any visible text.
    queries = list(observations.get("search_queries") or [])
    if question.strip():
        queries.insert(0, question.strip())
    texts = []
    for block in ([observations.get("image_a"), observations.get("image_b")] if compare else [observations]):
        for item in (block or {}).get("visible_text") or []:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
    queries.extend(texts[:3])
    if not queries:
        queries = [observations.get("subject_summary") or "historical photograph"]
    evidence = _gather_evidence(queries[:8], user_id, canvas_id, source_filters or [])

    # Citation labels in the order shown to the model.
    for index, row in enumerate(evidence, start=1):
        row["citation_id"] = index

    # 3. REASON
    task = (
        f"Compare image A ({image_names[0]}) and image B ({image_names[1]}): are they the same person, place or maker, "
        "and who / what / when are they?"
        if compare
        else f"Identify the image {image_names[0]}: who is depicted, who made it, when and where."
    )
    compare_shape = (
        '{"same_person_likelihood": 0-100 or null, "same_place_likelihood": 0-100 or null, '
        '"same_maker_likelihood": 0-100 or null, "shared_features": [], "differences": [], "verdict": ""}'
        if compare
        else "null"
    )
    reason_prompt = REASON_PROMPT.format(
        task=task,
        question_line=question_line,
        plural="s" if compare else "",
        observations=json.dumps(observations, ensure_ascii=False, indent=1)[:12000],
        evidence=build_context(evidence) if evidence else "No indexed passages matched. Base the report on the image only and say so.",
        compare_shape=compare_shape,
    )
    reason_text = generate_text(prompt=reason_prompt, system_instruction=REASON_SYSTEM, temperature=0.15, max_output_tokens=4096)
    report = _parse_json(reason_text) or {"summary": reason_text.strip()}

    # Tidy numbers so the interface can draw bars safely.
    candidates = []
    for item in report.get("candidates") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        item["likelihood"] = _clamp(item.get("likelihood")) or 0
        candidates.append(item)
    report["candidates"] = sorted(candidates, key=lambda item: item["likelihood"], reverse=True)[:8]
    if isinstance(report.get("comparison"), dict):
        for key in ("same_person_likelihood", "same_place_likelihood", "same_maker_likelihood"):
            report["comparison"][key] = _clamp(report["comparison"].get(key))

    return {
        "mode": "compare" if compare else "identify",
        "observations": observations,
        "report": report,
        "evidence": evidence,
    }

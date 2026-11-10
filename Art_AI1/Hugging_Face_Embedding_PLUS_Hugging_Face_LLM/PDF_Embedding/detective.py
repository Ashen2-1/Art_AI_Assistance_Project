"""
Photo Detective.

Modes (chosen by the number of images):
  identify  1 image       who / when / where / who made it
  compare   2 images      same person, place or maker?
  group     3-12 images   which images show the same person(s), and who
                          is each person?

The researcher can mark the person they care about in any image (a box,
plus an optional note like "the man on the left"). That region is cropped
and sent next to the full image, so the analysis focuses on that person.

Steps:
  1. LOOK     vision model: structured observations, people with
              approximate boxes, visible text, clues, search queries
              (and, for groups, clusters of the same person across images)
  2. SEARCH   the queries run against every indexed source in the
              workspace (books, PDFs, Word, sheets, other image descriptions)
  2b. WEB     optional ("Search the web for clues"): Gemini with Google
              Search looks for text clues (inscriptions, studios, places,
              events, captions) — never a person's appearance
  3. REASON   the answer model weighs observations against the evidence,
              cites [S1] / [W1], and gives calibrated likelihoods

Identity is never judged from facial resemblance: likelihoods that two
images show the same person, or that a person is a named candidate, rest
on documents and context. Features that change over time (age, facial
hair, hair, clothing…) are reported as consistent / not diagnostic.

The vision step also returns "markers": boxes on concrete visual details
(clothing, props, backdrops, inscriptions, stamps…) so the interface can
point at them on the photos. Faces are never marked.

Vision uses Gemini (the user's key when connected); reasoning follows the
user's answer provider (NEXO / Gemini / OpenAI / Claude).
"""

import json
import re
from typing import List, Optional

from gemini_client import generate_multimodal, generate_text, web_search
from rag_query import build_context, retrieve

MAX_EVIDENCE = 16
MAX_IMAGES = 12

LOOK_SYSTEM = """
You are a meticulous visual analyst for a historical image archive
(photographs, paintings, drawings, prints). You describe what is visible,
precisely and without inventing names. Return only valid JSON.
""".strip()

PERSON_KEYS = (
    '{"id": "P1", "position": "e.g. standing, second from left", "box": [ymin, xmin, ymax, xmax] (0-1000), '
    '"apparent_age": "", "sex": "", "clothing": "", "hair_and_face": "", "pose_and_action": "", "distinctive_features": ""}'
)

MARKER_KEYS = (
    '"markers": [{"id": "M1", "image": "A", "box": [ymin, xmin, ymax, xmax] (0-1000), '
    '"label": "short name of the detail", "kind": "shared | different | date | place | maker | text | object | condition", '
    '"group": "the same short key for the same detail seen in several images, else empty"}]'
)

MARKER_RULES = """
Markers: point at 3-10 concrete visual details that matter for dating, placing,
attributing or linking the images — clothing details, buttons, jewellery, uniforms,
backdrops, walls, furniture, props, buildings, inscriptions, studio stamps, mounts,
damage, signs of the photographic process. Boxes must tightly enclose the detail.
Never mark faces or facial features, and never use facial resemblance as a reason.
""".strip()

IMAGE_KEYS = f"""
  "medium": "photograph | oil painting | watercolour | drawing | sketch | print/engraving | other",
  "process_or_technique": "e.g. albumen print, calotype, stereograph card, oil on canvas, pencil",
  "subject_summary": "one or two sentences",
  "people": [{PERSON_KEYS}],
  "objects_and_setting": "",
  "visible_text": [{{"text": "exact transcription", "where": ""}}],
  "date_clues": [""],
  "place_clues": [""],
  "maker_clues": [""],
  "condition": "",
  "distinctive_features": [""]
""".strip()

LOOK_ONE = f"""
Examine the image and return JSON:
{{
{IMAGE_KEYS},
  "focus_person": "if a FOCUS crop is provided: which person in the full image it shows (id) and a detailed description, else empty",
  {MARKER_KEYS},
  "search_queries": ["4-6 short archive search queries: names or words from visible text, distinctive objects, place and era terms"]
}}
{MARKER_RULES}
""".strip()

LOOK_TWO = f"""
You are given image A and image B (each may be followed by a FOCUS crop of
the person the researcher cares about). Return JSON:
{{
  "image_a": {{ {IMAGE_KEYS} }},
  "image_b": {{ {IMAGE_KEYS} }},
  "comparison": {{
    "same_place_likelihood": 0-100 or null, "same_maker_likelihood": 0-100 or null,
    "shared_features": [""],
    "differences": ["differences that time alone cannot explain (e.g. different process, different building)"],
    "changes_with_time": ["differences that could simply come from years passing: age, facial hair, hair, weight, clothing, setting"],
    "person_context": "non-facial context that links or separates the people (same sitting, props, captions), or empty",
    "notes": ""
  }},
  {MARKER_KEYS},
  "search_queries": ["4-6 short archive search queries covering both images"]
}}
{MARKER_RULES}
In markers, give a detail seen in both images the same "group" key and kind "shared".
Do not estimate whether the people are the same person from their faces.
""".strip()

LOOK_GROUP = f"""
You are given {{count}} images labelled {{labels}} (an image may be followed by a
FOCUS crop of the person the researcher cares about). Some may show the same
person; some may show several people. Return JSON:
{{
  "images": [
    {{"label": "A", "medium": "", "subject_summary": "", "people": [{PERSON_KEYS}],
      "visible_text": [{{"text": "", "where": ""}}], "date_clues": [""], "place_clues": [""], "maker_clues": [""]}}
  ],
  "clusters": [
    {{"cluster_id": "C1", "members": ["A-P1", "C-P2"], "link": "the non-facial context that links them",
      "shared_features": ["specific matching details"], "doubts": ["differences"]}}
  ],
  {MARKER_KEYS},
  "search_queries": ["5-8 short archive search queries covering all images"]
}}
Person references are "<image label>-<person id>". Put every person in
exactly one cluster (a cluster may have one member). Only put people from
different images in one cluster when NON-FACIAL context links them: the same
sitting (same clothes, setting and props), a caption or inscription naming
them, or their place in a numbered series. Never group by facial resemblance.
Prefer focus persons as the first cluster.
{MARKER_RULES}
""".strip()

REASON_SYSTEM = """
You are a careful art and photo historian acting as a "detective". You
weigh visual observations against documentary evidence. Rules:
- Cite evidence with the labels you are given, written separately: [S1] [S3].
- Never invent sources, names, dates or quotations. If the evidence does
  not name anyone, say so and list what would be needed.
- Likelihoods are calibrated percentages (0-100). Several candidates may
  share the probability; leave room for "someone not in the sources".
- Separate what is SEEN in the image from what is INFERRED.
- If the researcher marked a focus person, the identification is about
  THAT person; mention others only as context.
- Never judge identity from facial resemblance. Likelihoods about WHO a
  person is (candidates, same person) rest only on documentary evidence
  (captions, inscriptions, names in the sources, credits, provenance, dated
  records, web pages) and context (same sitting, studio, props, events whose
  dates fit). Without such evidence keep them low and say what is missing.
  Give each likelihood a "basis": documentary | contextual | visual only.
- Features that change with time — apparent age, facial hair, hair length
  and colour, weight, clothing, setting, colour vs black-and-white — are NOT
  evidence against two images showing the same person or place when they
  may be years apart. Put them under "consistent_not_diagnostic", never
  under evidence against. Evidence against must be a fixed fact (documented
  identities, dates or places that cannot both be true).
- If the researcher states a hypothesis (e.g. "the same man at different
  ages"), test it explicitly in "hypothesis_check".
- Web results [W#] may be wrong or about another subject: prefer primary
  sources (archives, museums, publishers, newspapers) and say how reliable
  each is.
Return only valid JSON.
""".strip()

CANDIDATE_SHAPE = (
    '{"name": "", "kind": "sitter | maker | place | object | other", "likelihood": 0-100, '
    '"basis": "documentary | contextual | visual only", '
    '"evidence_for": ["reason with citation [S#] or [W#]"], "evidence_against": ["fixed facts only"]}'
)

HYPOTHESIS_SHAPE = (
    '{"hypothesis": "the researcher\'s hypothesis in one line, or empty if none", '
    '"verdict": "supported | consistent but unproven | contradicted | cannot tell", '
    '"supports": ["documentary or contextual evidence for it, with citations"], '
    '"consistent_not_diagnostic": ["observations that fit it but prove nothing (e.g. beard grown, older)"], '
    '"contradicts": ["fixed facts against it, with citations"], '
    '"would_settle": ["the evidence that would decide it"]}'
)

REASON_SINGLE = """
Task: {task}
{focus_line}
{question_line}

VISUAL OBSERVATIONS:
{observations}

EVIDENCE FROM THE WORKSPACE SOURCES:
{evidence}
{web_evidence}

Return JSON:
{{
  "summary": "3-5 sentences with citations",
  "hypothesis_check": {hypothesis},
  "candidates": [{candidate}],
  "unidentified": "if no candidate is supported: what the image shares with the sources and what would identify it, else empty",
  "date_estimate": {{"range": "", "confidence": "low | medium | high", "reasoning": ""}},
  "place_estimate": {{"value": "", "confidence": "low | medium | high", "reasoning": ""}},
  "maker_estimate": {{"value": "", "confidence": "low | medium | high", "reasoning": ""}},
  "similar_items": [{{"source": "file name", "citation": "S#", "shared_features": ""}}],
  "comparison": {compare_shape},
  "transcribed_text": [""],
  "missing_evidence": ["what would raise or lower the likelihoods"],
  "open_questions": [""],
  "next_steps": [""]
}}
""".strip()

REASON_GROUP = """
Task: {count} images ({labels}). Work out which images show the same
person and who each person is.
{focus_line}
{question_line}

VISUAL OBSERVATIONS (with proposed clusters of the same person):
{observations}

EVIDENCE FROM THE WORKSPACE SOURCES:
{evidence}
{web_evidence}

Return JSON:
{{
  "summary": "4-6 sentences: how many distinct people, who they probably are, with citations",
  "hypothesis_check": {hypothesis},
  "people": [
    {{"cluster_id": "C1", "members": ["A-P1", "C-P2"], "same_person_likelihood": 0-100 (documentary / contextual links only),
      "label": "short description, e.g. bearded man in frock coat",
      "candidates": [{candidate}],
      "shared_features": [""], "doubts": [""]}}
  ],
  "date_estimate": {{"range": "", "confidence": "low | medium | high", "reasoning": ""}},
  "place_estimate": {{"value": "", "confidence": "low | medium | high", "reasoning": ""}},
  "maker_estimate": {{"value": "", "confidence": "low | medium | high", "reasoning": ""}},
  "similar_items": [{{"source": "file name", "citation": "S#", "shared_features": ""}}],
  "transcribed_text": [""],
  "missing_evidence": [""],
  "open_questions": [""],
  "next_steps": [""]
}}
""".strip()

WEB_SYSTEM = """
You research historical photographs, paintings and drawings on the web for
an archive researcher. Search only TEXT clues: words and names visible in the
image, names already given by the researcher or the sources, studios,
photographers, publishers, places, buildings, events, objects and dates.
Never search by describing a person's face or body, and never try to name a
person from their appearance. Report only what pages actually say.
""".strip()

WEB_PROMPT = """
Find documentary clues on the web for this investigation.
{question_line}

What the images show (from a visual analysis):
{clues}

Write 4-8 short findings. Each: what the page says and why it matters
(who made or published the image, where it is held, a caption, a date, a
place, a related event). Say plainly if nothing relevant was found.
""".strip()


def _parse_json(text: str) -> Optional[dict]:
    """Models sometimes wrap JSON in prose or code fences."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{"): cleaned.rfind("}") + 1] if "{" in cleaned else ""):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except ValueError:
            continue
    return None


def _clamp(value):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def _clean_box(box):
    """[ymin, xmin, ymax, xmax] in 0-1000, or None."""
    try:
        values = [max(0, min(1000, int(round(float(v))))) for v in box][:4]
        if len(values) == 4 and values[2] > values[0] and values[3] > values[1]:
            return values
    except (TypeError, ValueError):
        pass
    return None


def _tidy_candidates(items):
    out = []
    for item in items or []:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            item["likelihood"] = _clamp(item.get("likelihood")) or 0
            out.append(item)
    return sorted(out, key=lambda item: item["likelihood"], reverse=True)[:8]


def _gather_evidence(queries: List[str], user_id: str, canvas_id: str, source_filters: List[str]) -> List[dict]:
    seen = {}
    for query in queries:
        query = str(query or "").strip()
        if not query:
            continue
        try:
            rows = retrieve(question=query, user_id=user_id, canvas_id=canvas_id, top_k=4, source_filters=source_filters or None)
        except Exception as error:  # one failed query should not stop the analysis
            print(f"[Detective] search failed for {query!r}: {error}")
            continue
        for row in rows:
            key = (row["source"], row["chunk_idx"])
            if key not in seen or row["score"] > seen[key]["score"]:
                seen[key] = row
    best = sorted(seen.values(), key=lambda row: row["score"], reverse=True)[:MAX_EVIDENCE]
    for index, row in enumerate(best, start=1):
        row["citation_id"] = index
    return best


MARKER_KINDS = {"shared", "different", "date", "place", "maker", "text", "object", "condition"}


def _clean_markers(observations: dict, labels: List[str]) -> List[dict]:
    """Boxes the interface draws on the photos (never faces — see MARKER_RULES)."""
    out = []
    for index, item in enumerate(observations.get("markers") or []):
        if not isinstance(item, dict):
            continue
        box = _clean_box(item.get("box"))
        image = str(item.get("image") or labels[0]).strip().upper()[:1]
        label = str(item.get("label") or "").strip()[:80]
        if not box or image not in labels or not label:
            continue
        kind = str(item.get("kind") or "").strip().lower()
        out.append({
            "id": f"M{len(out) + 1}",
            "image": image,
            "box": box,
            "label": label,
            "kind": kind if kind in MARKER_KINDS else "object",
            "group": str(item.get("group") or "").strip().lower()[:40],
        })
        if len(out) >= 24:
            break
    return out


def _web_clues(observations: dict, blocks: List[dict], question: str, focus_line: str) -> dict:
    """Optional web step: text clues only (see WEB_SYSTEM)."""
    lines = []
    for block in blocks:
        block = block or {}
        for key, title in (("subject_summary", "Subject"), ("medium", "Medium"), ("process_or_technique", "Process"), ("objects_and_setting", "Setting")):
            if block.get(key):
                lines.append(f"- {title}: {str(block[key])[:300]}")
        texts = [str(item.get("text")) for item in block.get("visible_text") or [] if isinstance(item, dict) and item.get("text")]
        if texts:
            lines.append("- Visible text: " + " | ".join(texts)[:400])
        for key, title in (("date_clues", "Date clues"), ("place_clues", "Place clues"), ("maker_clues", "Maker clues")):
            values = [str(value) for value in block.get(key) or [] if value]
            if values:
                lines.append(f"- {title}: " + "; ".join(values)[:300])
    queries = [str(query) for query in observations.get("search_queries") or []][:6]
    if queries:
        lines.append("- Suggested searches: " + "; ".join(queries))
    question_line = "\n".join(part for part in (f"The researcher asks: {question.strip()}" if question.strip() else "", focus_line) if part)
    return web_search(WEB_PROMPT.format(question_line=question_line, clues="\n".join(lines)[:5000] or "- (no clues)"), system_instruction=WEB_SYSTEM)


def _labels(count: int) -> List[str]:
    return [chr(ord("A") + index) for index in range(count)]


def analyze(
    images: List[dict],
    user_id: str,
    canvas_id: str = "default",
    question: str = "",
    source_filters: Optional[List[str]] = None,
    web: bool = False,
) -> dict:
    """
    images: [{"path", "name", "focus_path" (optional crop), "focus_note"}]
    web: also search the web for text clues (inscriptions, studios, places…).
    """
    images = images[:MAX_IMAGES]
    count = len(images)
    mode = "identify" if count == 1 else "compare" if count == 2 else "group"
    labels = _labels(count)
    question_line = f"The researcher asks: {question.strip()}" if question.strip() else ""

    # Images in order; a FOCUS crop follows its image.
    paths, legend, focus_lines = [], [], []
    for label, image in zip(labels, images):
        paths.append(image["path"])
        legend.append(f"Image {label}: {image['name']}")
        if image.get("focus_path"):
            paths.append(image["focus_path"])
            note = f' ("{image["focus_note"]}")' if image.get("focus_note") else ""
            legend.append(f"Image {label} FOCUS crop: the person/region the researcher wants identified{note}")
            focus_lines.append(f"Image {label}: the researcher marked a focus person{note}.")
        elif image.get("focus_note"):
            focus_lines.append(f'Image {label}: the researcher is interested in "{image["focus_note"]}".')
    focus_line = "\n".join(focus_lines)
    header = "Images in this request, in order:\n" + "\n".join(legend)

    # 1. LOOK
    if mode == "identify":
        look_prompt = LOOK_ONE
    elif mode == "compare":
        look_prompt = LOOK_TWO
    else:
        look_prompt = LOOK_GROUP.replace("{count}", str(count)).replace("{labels}", ", ".join(labels))
    look_prompt = "\n\n".join(part for part in (header, focus_line, question_line, look_prompt) if part)
    look_text = generate_multimodal(look_prompt, paths, system_instruction=LOOK_SYSTEM, temperature=0.1, max_output_tokens=8192, json_output=True)
    observations = _parse_json(look_text) or {"subject_summary": look_text.strip()}

    # Clean person boxes so the interface can draw face crops safely.
    blocks = (
        [observations]
        if mode == "identify"
        else [observations.get("image_a") or {}, observations.get("image_b") or {}]
        if mode == "compare"
        else observations.get("images") or []
    )
    for block in blocks:
        for person in (block or {}).get("people") or []:
            if isinstance(person, dict):
                person["box"] = _clean_box(person.get("box"))

    # 2. SEARCH: model queries, the question, focus notes and visible text.
    queries = list(observations.get("search_queries") or [])
    if question.strip():
        queries.insert(0, question.strip())
    queries.extend(image["focus_note"] for image in images if image.get("focus_note"))
    for block in blocks:
        for item in (block or {}).get("visible_text") or []:
            if isinstance(item, dict) and item.get("text"):
                queries.append(str(item["text"]))
    if not queries:
        queries = [observations.get("subject_summary") or "historical photograph"]
    evidence = _gather_evidence(queries[:10], user_id, canvas_id, source_filters or [])
    evidence_text = build_context(evidence) if evidence else "No indexed passages matched. Base the report on the images only and say so."
    markers = _clean_markers(observations, labels)
    observations.pop("markers", None)
    observations_text = json.dumps(observations, ensure_ascii=False, indent=1)[:16000]

    # 2b. WEB (optional): documentary clues from the web, cited as [W1], [W2]…
    web_sources, web_note, web_queries = [], "", []
    if web:
        try:
            found = _web_clues(observations, blocks, question, focus_line)
            web_queries = found.get("queries") or []
            for index, item in enumerate(found.get("sources") or [], start=1):
                web_sources.append({"citation_id": f"W{index}", "title": item["title"], "url": item["url"], "text": item.get("text") or ""})
            summary = (found.get("text") or "").strip()
            web_lines = [f"[{item['citation_id']}] {item['title']} — {item['url']}\n{item['text'] or '(no excerpt)'}" for item in web_sources]
            web_evidence = (
                "EVIDENCE FROM THE WEB (cite as [W#]):\n" + (summary[:3000] + "\n\n" if summary else "") + "\n\n".join(web_lines)
                if web_sources or summary
                else "EVIDENCE FROM THE WEB: the web search found nothing relevant."
            )
            if not web_sources:
                web_note = "The web search found no relevant pages."
        except Exception as error:  # the report still works without the web
            print(f"[Detective] web search failed: {error}")
            web_note = "The web search could not run this time, so the report uses your sources only."
            web_evidence = "EVIDENCE FROM THE WEB: not available."
    else:
        web_evidence = ""
    hypothesis = HYPOTHESIS_SHAPE if question.strip() else "null"

    # 3. REASON
    if mode == "group":
        reason_prompt = REASON_GROUP.format(
            count=count,
            labels=", ".join(f"{label} = {image['name']}" for label, image in zip(labels, images)),
            focus_line=focus_line,
            question_line=question_line,
            observations=observations_text,
            evidence=evidence_text,
            web_evidence=web_evidence,
            hypothesis=hypothesis,
            candidate=CANDIDATE_SHAPE,
        )
    else:
        task = (
            f"Compare image A ({images[0]['name']}) and image B ({images[1]['name']}): same person, place or maker? Who / what / when?"
            if mode == "compare"
            else f"Identify the image {images[0]['name']}: who is depicted, who made it, when and where."
        )
        compare_shape = (
            '{"same_person_likelihood": 0-100 or null (documentary / contextual evidence only, never faces), '
            '"same_place_likelihood": 0-100 or null, "same_maker_likelihood": 0-100 or null, '
            '"basis": "what the likelihoods rest on", "shared_features": [], '
            '"differences": ["only differences time cannot explain"], '
            '"consistent_not_diagnostic": ["differences that years passing can explain"], '
            '"missing_evidence": [""], "verdict": ""}'
            if mode == "compare"
            else "null"
        )
        reason_prompt = REASON_SINGLE.format(
            task=task,
            focus_line=focus_line,
            question_line=question_line,
            observations=observations_text,
            evidence=evidence_text,
            web_evidence=web_evidence,
            hypothesis=hypothesis,
            candidate=CANDIDATE_SHAPE,
            compare_shape=compare_shape,
        )
    reason_text = generate_text(prompt=reason_prompt, system_instruction=REASON_SYSTEM, temperature=0.15, max_output_tokens=6144)
    report = _parse_json(reason_text) or {"summary": reason_text.strip()}

    # Tidy numbers for the interface.
    report["candidates"] = _tidy_candidates(report.get("candidates"))
    if isinstance(report.get("comparison"), dict):
        for key in ("same_person_likelihood", "same_place_likelihood", "same_maker_likelihood"):
            report["comparison"][key] = _clamp(report["comparison"].get(key))
    people = []
    for person in report.get("people") or []:
        if not isinstance(person, dict):
            continue
        person["candidates"] = _tidy_candidates(person.get("candidates"))
        person["same_person_likelihood"] = _clamp(person.get("same_person_likelihood"))
        person["members"] = [str(member) for member in person.get("members") or []][:MAX_IMAGES * 4]
        people.append(person)
    report["people"] = people
    if not isinstance(report.get("hypothesis_check"), dict) or not str(report["hypothesis_check"].get("verdict") or "").strip():
        report["hypothesis_check"] = None
    if web_note:
        report["web_note"] = web_note

    return {
        "mode": mode,
        "labels": labels,
        "observations": observations,
        "markers": markers,
        "report": report,
        "evidence": evidence,
        "web": web_sources,
        "web_queries": web_queries,
    }

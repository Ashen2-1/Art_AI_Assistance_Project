"""
Canvas Agent: turn a request ("make a comparison table of these three papers
and group them by theme") into a short plan of canvas actions. The browser
shows the plan; nothing runs until the user presses Execute, and every plan
can be undone as one step.

The model only proposes actions from a fixed list; the browser checks every
step again before running it.
"""

import json
import re
from typing import Optional

from gemini_client import generate_text

ACTIONS = """
Allowed actions (JSON objects; "cards" are card ids from the list, or "step:N" for a card created by step N):
- {"action": "create_note", "title": str, "text": str (Markdown; maths in $...$), "near": card?}
- {"action": "create_table", "title": str, "columns": [str], "rows": [[str]], "near": card?}
- {"action": "group", "cards": [card, ...] (2+), "name": str}
- {"action": "link", "from": card, "to": card, "label": str?}
- {"action": "run_tool", "tool": "summary"|"compare"|"evidence"|"gaps"|"outline"|"quiz", "cards": [card, ...]}
- {"action": "side_by_side", "cards": [card, ...] (2-8)}
- {"action": "select", "cards": [card, ...]}
- {"action": "arrange"}   (tidy the whole board)
Every step also has "why": a short reason the user will read.
""".strip()

SYSTEM = (
    "You are the planning part of NEXO, a research canvas. You turn the user's request into a short, "
    "safe plan of canvas actions. Never delete anything. Use only the listed actions and only card ids "
    "that appear in the canvas list (or step:N for cards your plan creates). Prefer few steps (at most 10). "
    "When the user asks for content (a table, a summary note, an outline), write that content yourself "
    "from the card text you were given — do not invent facts that are not in the cards; mark gaps as '?'. "
    "If the request cannot be done with these actions, return no steps and explain in the summary. "
    "Answer with JSON only."
)


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def plan_canvas_actions(request: str, canvas: dict, history: Optional[list] = None, preferences: str = "") -> dict:
    cards = canvas.get("cards") or []
    groups = canvas.get("groups") or []
    lines = []
    for card in cards[:120]:
        flags = []
        if card.get("selected"):
            flags.append("SELECTED")
        if card.get("group"):
            flags.append(f"group={card.get('group')}")
        snippet = str(card.get("text") or "").replace("\n", " ")[:400]
        lines.append(f"- id={card.get('id')} | {card.get('type')} | \"{card.get('title')}\" {' '.join(flags)}\n  {snippet}")
    group_lines = [f"- id={g.get('id')} name=\"{g.get('name')}\" cards={g.get('count')}" for g in groups[:40]]
    recent = "\n".join(f"{m.get('role')}: {str(m.get('text') or '')[:400]}" for m in (history or [])[-6:])

    prompt = f"""{ACTIONS}

Canvas cards ({len(cards)}):
{chr(10).join(lines) or "(empty canvas)"}

Groups:
{chr(10).join(group_lines) or "(none)"}

{"The user's saved preferences (follow them when they apply):" + chr(10) + preferences.strip()[:2000] if preferences.strip() else ""}

Recent conversation:
{recent or "(none)"}

User request:
{request.strip()[:2000]}

If the user says "these" or "the selected ones", use the SELECTED cards.
Return JSON: {{"summary": "one or two sentences: what you will do", "steps": [ ... ]}}
Write the summary and every "why" in the language of the request."""

    text = generate_text(prompt=prompt, system_instruction=SYSTEM, temperature=0.2, max_output_tokens=6000)
    plan = _extract_json(text) or {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    return {
        "summary": str(plan.get("summary") or ("I could not make a plan for that." if not steps else "")).strip()[:600],
        "steps": steps[:12],
    }

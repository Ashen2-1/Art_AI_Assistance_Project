########################## rag_query.py
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional, List
import json

# Ensure files in this folder can be imported
sys.path.insert(0, str(Path(__file__).parent))

from gemini_embedding import embed_query
from db import search_chunks, get_stats
from gemini_client import generate_text, generate_image
from prompt_profiles import build_system_prompt

TOP_K = 3


# This instruction is intentionally domain-neutral.
# A project-specific domain will be added later.
RESEARCH_SYSTEM_INSTRUCTION = """
You are NEXO, a careful multidisciplinary research assistant.

You may assist with art, humanities, engineering, mathematics,
science, technology, and other research fields.

Adapt your terminology and explanation to the user's question
and the supplied research materials. Do not assume that every
project is about art history.

When source context is supplied:

1. Base factual claims only on the supplied context.
2. Clearly distinguish source-supported statements from inference.
3. Mention the source filename when useful.
4. If the answer is not supported by the context, say:
   "I cannot find this in the provided sources."
5. Do not invent quotations, page numbers, citations, results,
   formulas, or evidence.

Use clear Markdown with short paragraphs. Use bullet points when
they make the response easier to understand.
""".strip()


GENERAL_SYSTEM_INSTRUCTION = """
You are NEXO, a helpful multidisciplinary research assistant.

You can help with art, humanities, engineering, mathematics,
science, technology, and other research fields.

Adapt to the user's subject and level of knowledge. Explain ideas
clearly, accurately, and concisely. Do not fabricate sources,
results, formulas, quotations, or citations.

Use Markdown and short readable paragraphs.
""".strip()


IMAGE_SYSTEM_INSTRUCTION = """
You are NEXO, a multidisciplinary visual research assistant.

Analyze the supplied image carefully. The image may contain an
artwork, diagram, graph, engineering component, mathematical work,
scientific material, document page, or other research content.

Describe only what can reasonably be observed. Clearly label
uncertain interpretations. Do not invent names, dates, measurements,
equations, sources, or historical facts that are not supported by
the image or supplied context.
""".strip()


def format_chat_history(chat_history: list) -> str:
    if not chat_history:
        return "No previous conversation."

    lines = []

    for message in chat_history[-8:]:
        role = message.get("role", "user")
        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            continue

        role_label = (
            "ASSISTANT"
            if role in {"ai", "assistant"}
            else "USER"
        )

        lines.append(
            f"{role_label}: {content}"
        )

    return (
        "\n".join(lines)
        if lines
        else "No previous conversation."
    )


def normalize_source_filters(
    source_filters: Optional[List[str]] = None,
    source_filter: Optional[str] = None,
) -> List[str]:
    """
    Support both the new multiple-source format and the old
    single-source format during the migration.
    """

    candidates = list(source_filters or [])

    if source_filter:
        candidates.append(source_filter)

    cleaned_sources = []
    seen_sources = set()

    for source in candidates:
        normalized_source = str(source).strip()

        if (
            normalized_source
            and normalized_source not in seen_sources
        ):
            seen_sources.add(normalized_source)
            cleaned_sources.append(normalized_source)

    return cleaned_sources


def retrieve(
    question: str,
    user_id: str,
    canvas_id: str = "default",
    top_k: int = TOP_K,
    source_filters: Optional[List[str]] = None,
    source_filter: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve chunks only from the current user's Canvas.

    When multiple sources are selected, retrieve relevant chunks
    from every selected source. This prevents one document from
    taking all available result positions.
    """

    if not user_id or not str(user_id).strip():
        raise ValueError(
            "user_id is required when retrieving document chunks."
        )

    safe_user_id = str(user_id).strip()
    safe_canvas_id = (
        str(canvas_id or "default").strip()
        or "default"
    )
    safe_top_k = max(1, min(int(top_k), 10))

    selected_sources = normalize_source_filters(
        source_filters=source_filters,
        source_filter=source_filter,
    )

    stats = get_stats(
        user_id=safe_user_id,
        canvas_id=safe_canvas_id,
    )

    if stats["total_chunks"] == 0:
        print(
            "[RAG] This user's Canvas has no indexed chunks."
        )
        return []

    query_vector = embed_query(question)

    rows = []

    if selected_sources:
        # Retrieve from every selected document separately.
        # Therefore two selected documents cannot result in
        # chunks coming from only one document.
        for selected_source in selected_sources:
            source_rows = search_chunks(
                query_embedding=query_vector,
                user_id=safe_user_id,
                canvas_id=safe_canvas_id,
                top_k=safe_top_k,
                source_filters=[selected_source],
            )

            rows.extend(source_rows)
    else:
        rows = search_chunks(
            query_embedding=query_vector,
            user_id=safe_user_id,
            canvas_id=safe_canvas_id,
            top_k=safe_top_k,
            source_filters=None,
        )

    # Put the strongest results first while preserving results
    # from every selected source.
    rows.sort(
        key=lambda row: float(row["score"]),
        reverse=True,
    )

    chunks = []

    for citation_id, row in enumerate(rows, start=1):
        chunks.append(
            {
                "citation_id": citation_id,
                "text": row["content"],
                "source": row["source"],
                "chunk_idx": row["chunk_index"],
                "score": round(
                    float(row["score"]),
                    4,
                ),
            }
        )

    return chunks


def build_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No source context was retrieved."

    return "\n\n".join(
        (
            f"[{chunk['citation_id']}] "
            f"Source: {chunk['source']}, "
            f"chunk {chunk['chunk_idx'] + 1}\n"
            f"{chunk['text']}"
        )
        for chunk in chunks
    )


def query_text_rag(
    question: str,
    user_id: str,
    canvas_id: str = "default",
    top_k: int = 5,
    source_filters=None,
    source_filter=None,
    chat_history=None,
):
    safe_user_id = str(user_id or "").strip()

    if not safe_user_id:
        raise ValueError(
            "user_id is required for text RAG."
        )

    safe_canvas_id = (
        str(canvas_id or "default").strip()
        or "default"
    )

    safe_top_k = max(
        1,
        min(int(top_k or 5), 10),
    )

    selected_sources = []

    if isinstance(source_filters, list):
        selected_sources.extend(source_filters)

    elif isinstance(source_filters, str) and source_filters.strip():
        try:
            parsed_sources = json.loads(source_filters)

            if isinstance(parsed_sources, list):
                selected_sources.extend(parsed_sources)
            else:
                selected_sources.append(source_filters)
        except json.JSONDecodeError:
            selected_sources.append(source_filters)

    if source_filter:
        selected_sources.append(source_filter)

    selected_sources = list(
        dict.fromkeys(
            str(source).strip()
            for source in selected_sources
            if str(source).strip()
        )
    )
    """
    Answer using only documents belonging to the current user
    and Canvas.
    """

    print(
        f"[RAG] Text RAG question: {question!r}"
    )

    selected_sources = normalize_source_filters(
        source_filters=source_filters,
        source_filter=source_filter,
    )

    chunks = retrieve(
        top_k=safe_top_k,
        user_id=safe_user_id,
        canvas_id=safe_canvas_id,
        source_filters=selected_sources,
    )

    if not chunks:
        return {
            "answer": (
                "No matching source material was found in this "
                "Canvas. Please upload or select a document first."
            ),
            "chunks": [],
            "mode": "text_rag",
        }

    context = build_context(chunks)

    history = format_chat_history(
        chat_history or []
    )

    selected_source_text = (
        "\n".join(
            f"- {source}"
            for source in selected_sources
        )
        if selected_sources
        else "All indexed sources in the current Canvas"
    )

    prompt = f"""
Use the retrieved source context to answer the user's research
question.

Selected sources:
{selected_source_text}

Conversation history:
<conversation_history>
{history}
</conversation_history>

Retrieved source context:
<context>
{context}
</context>

Current user question:
{question}

Requirements:

1. Consider every selected source when more than one source is
   selected.
2. Cite factual claims using the matching citation number, such
   as [1] or [2].
3. Do not cite a number that does not exist in the retrieved
   context.
4. If selected sources disagree, explain the disagreement.
5. If the sources do not contain enough evidence, state the
   limitation clearly.
6. Use the conversation history only to understand follow-up
   questions. Do not treat conversation history as source evidence.
7. If mathematical notation appears in the source, preserve it using standard LaTeX inside $...$.
   If the answer is not about math or formulas, avoid unnecessary LaTeX notation.
""".strip()

    answer = generate_text(
        prompt=prompt,
        system_instruction=RESEARCH_SYSTEM_INSTRUCTION,
        temperature=0.15,
        max_output_tokens=2048,
    )

    return {
        "answer": answer,
        "chunks": chunks,
        "mode": "text_rag",
    }

def query_general(
    question: str,
    chat_history: list = None,
    domain: str = "auto",
    task: str = "answer",
) -> dict:
    """
    General chat without document retrieval.
    """

    print(
        f"[RAG] General question: {question!r}"
    )

    history = format_chat_history(
        chat_history or []
    )

    prompt = f"""
Conversation history:
<conversation_history>
{history}
</conversation_history>

Current user question:
{question}
""".strip()

    answer = generate_text(
        prompt=prompt,
        system_instruction=GENERAL_SYSTEM_INSTRUCTION,
        temperature=0.3,
        max_output_tokens=2048,
    )

    return {
        "answer": answer,
        "mode": "general_chat",
        "chunks": [],
    }


def query_vlm_only(
    question: str,
    image_path: str,
) -> dict:
    """
    Analyze an image without document retrieval.
    """

    print(
        f"[RAG] Image analysis: {image_path}"
    )

    prompt = f"""
Analyze the supplied image and answer this question:

{question}

If important information cannot be determined from the image,
state the limitation clearly.
""".strip()

    answer = generate_image(
        prompt=prompt,
        image_path=image_path,
        system_instruction=IMAGE_SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=2048,
    )

    return {
        "answer": answer,
        "mode": "vlm_only",
        "image": image_path,
        "chunks": [],
    }


def query_hybrid(
    question: str,
    image_path: str,
    user_id: str,
    canvas_id: str = "default",
    top_k: int = TOP_K,
    source_filters: Optional[List[str]] = None,
    source_filter: Optional[str] = None,
    chat_history: list = None,
) -> dict:
    """
    Analyze an image together with document context belonging
    only to the current user and Canvas.
    """

    print(
        f"[RAG] Hybrid query: {question!r}"
    )

    chunks = retrieve(
        question=question,
        user_id=user_id,
        canvas_id=canvas_id,
        top_k=top_k,
        source_filters=source_filters,
        source_filter=source_filter,
    )

    context = build_context(chunks)

    history = format_chat_history(
        chat_history or []
    )

    prompt = f"""
Analyze the supplied image together with the retrieved research
context.

Conversation history:
<conversation_history>
{history}
</conversation_history>

Retrieved source context:
<context>
{context}
</context>

Current user question:
{question}

Clearly distinguish:

- What is visible in the image
- What is supported by the supplied sources
- What is interpretation or inference

Use citations such as [1] and [2] for claims supported by the
retrieved source context.
""".strip()

    answer = generate_image(
        prompt=prompt,
        image_path=image_path,
        system_instruction=(
            IMAGE_SYSTEM_INSTRUCTION
            + "\n\n"
            + RESEARCH_SYSTEM_INSTRUCTION
        ),
        temperature=0.15,
        max_output_tokens=2048,
    )

    return {
        "answer": answer,
        "chunks": chunks,
        "mode": "hybrid",
        "image": image_path,
    }


def print_result(result: dict):
    print("\n" + "=" * 65)
    print(
        "MODE:",
        result["mode"].upper().replace("_", " "),
    )
    print("=" * 65)

    print("\nANSWER:\n")

    for line in result["answer"].splitlines():
        if line.strip():
            print(
                textwrap.fill(
                    line,
                    width=80,
                )
            )
        else:
            print()

    if result.get("chunks"):
        print("\nRETRIEVED SOURCES:")

        for index, chunk in enumerate(
            result["chunks"],
            start=1,
        ):
            print(
                f"\n[{index}] "
                f"{chunk['source']} | "
                f"chunk {chunk['chunk_idx'] + 1} | "
                f"similarity={chunk['score']}"
            )

    if result.get("image"):
        print(
            f"\nImage: {result['image']}"
        )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except AttributeError:
        pass

    if len(sys.argv) < 3:
        print("Usage:")
        print(
            'python rag_query.py general '
            '"Explain machine learning."'
        )
        print(
            'python rag_query.py text '
            '"What does this source argue?"'
        )
        print(
            'python rag_query.py vlm '
            '"Analyze this image." image.jpg'
        )
        print(
            'python rag_query.py hybrid '
            '"Connect the image to the sources." image.jpg'
        )
        sys.exit(1)

    mode = sys.argv[1].lower()
    question = sys.argv[2]

    cli_user_id = os.getenv(
        "NEXO_CLI_USER_ID",
        "local-cli",
    )

    cli_canvas_id = os.getenv(
        "NEXO_CLI_CANVAS_ID",
        "default",
    )

    if mode == "general":
        result = query_general(question)


    elif mode == "text":

        result = query_text_rag(
            question=question,
            user_id=cli_user_id,
            canvas_id=cli_canvas_id,
        )

    elif mode == "vlm":
        if len(sys.argv) < 4:
            print(
                "VLM mode requires an image path."
            )
            sys.exit(1)

        result = query_vlm_only(
            question,
            sys.argv[3],
        )

    elif mode == "hybrid":
        if len(sys.argv) < 4:
            print(
                "Hybrid mode requires an image path."
            )
            sys.exit(1)

        result = query_hybrid(
            question=question,
            image_path=sys.argv[3],
            user_id=cli_user_id,
            canvas_id=cli_canvas_id,
        )

    else:
        print(
            "Unknown mode. Use: "
            "general | text | vlm | hybrid"
        )
        sys.exit(1)

    print_result(result)
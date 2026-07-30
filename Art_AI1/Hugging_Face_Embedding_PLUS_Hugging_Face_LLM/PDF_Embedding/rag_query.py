import sys
import textwrap
from pathlib import Path
from typing import Optional, List

# Ensure files in this folder can be imported
sys.path.insert(0, str(Path(__file__).parent))

from gemini_embedding import embed_query
from db import search_chunks, get_stats
from gemini_client import generate_text, generate_image


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


def retrieve(
    question: str,
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve relevant chunks from PostgreSQL using the
    existing BGE embedding model.

    This embedding model will be replaced in the next stage.
    """

    stats = get_stats()

    if stats["total_chunks"] == 0:
        print(
            "[RAG] Database is empty. "
            "Please ingest a document first."
        )
        return []

    query_vector = embed_query(
        question
    )

    rows = search_chunks(
        query_vector,
        top_k=top_k,
        source_filter=source_filter,
    )

    return [
        {
            "text": row["content"],
            "source": row["source"],
            "chunk_idx": row["chunk_index"],
            "score": round(
                float(row["score"]),
                4,
            ),
        }
        for row in rows
    ]


def build_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No source context was retrieved."

    return "\n\n".join(
        (
            f"[Source: {chunk['source']}, "
            f"chunk {chunk['chunk_idx'] + 1}]\n"
            f"{chunk['text']}"
        )
        for chunk in chunks
    )


def query_text_rag(
    question: str,
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
    chat_history: list = None,
) -> dict:
    """
    Ask a question using retrieved document context.
    """

    print(
        f"[RAG] Text RAG question: {question!r}"
    )

    chunks = retrieve(
        question,
        top_k=top_k,
        source_filter=source_filter,
    )

    if not chunks:
        return {
            "answer": (
                "No matching source material was found. "
                "Please upload or select a document first."
            ),
            "chunks": [],
            "mode": "text_rag",
        }

    context = build_context(chunks)
    history = format_chat_history(
        chat_history or []
    )

    prompt = f"""
Use the source context to answer the user's research question.

Selected source filter:
{source_filter or "All available sources"}

Conversation history:
<conversation_history>
{history}
</conversation_history>

Retrieved source context:
<context>
{context}
</context>

User question:
{question}
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
    top_k: int = TOP_K,
    source_filter: Optional[str] = None,
) -> dict:
    """
    Analyze an image together with retrieved source context.
    """

    print(
        f"[RAG] Hybrid query: {question!r}"
    )

    chunks = retrieve(
        question,
        top_k=top_k,
        source_filter=source_filter,
    )

    context = build_context(chunks)

    prompt = f"""
Analyze the supplied image together with the retrieved research
context.

Retrieved source context:
<context>
{context}
</context>

User question:
{question}

Clearly distinguish:

- What is visible in the image
- What is supported by the supplied sources
- What is an interpretation or inference
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

    if mode == "general":
        result = query_general(question)

    elif mode == "text":
        result = query_text_rag(question)

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
            question,
            sys.argv[3],
        )

    else:
        print(
            "Unknown mode. Use: "
            "general | text | vlm | hybrid"
        )
        sys.exit(1)

    print_result(result)
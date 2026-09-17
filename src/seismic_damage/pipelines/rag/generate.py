"""Answer generation that may only use retrieved evidence."""

from __future__ import annotations

from seismic_damage.schemas.pipeline import RAGDocument
from seismic_damage.schemas.rag import RAGCitation


def evidence_is_sufficient(
    documents: list[RAGDocument],
    *,
    min_score: float,
    refuse_if_insufficient: bool,
) -> bool:
    if not documents:
        return False
    if not refuse_if_insufficient:
        return True
    return documents[0].score >= min_score and bool(documents[0].content.strip())


def citations_from_documents(documents: list[RAGDocument], limit: int = 5) -> list[RAGCitation]:
    citations: list[RAGCitation] = []
    for doc in documents[:limit]:
        excerpt = doc.content.strip().replace("\n", " ")
        if len(excerpt) > 280:
            excerpt = excerpt[:277] + "..."
        citations.append(
            RAGCitation(
                document_id=doc.document_id,
                chunk_id=str(doc.metadata.get("chunk_id") or doc.document_id),
                page_number=doc.metadata.get("page_number"),
                score=doc.score,
                excerpt=excerpt,
                source_path=doc.source_path,
            )
        )
    return citations


def extractive_answer(query: str, documents: list[RAGDocument]) -> str:
    lines = [f"Answer grounded in retrieved report chunks for query: {query}", ""]
    for index, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page_number")
        chunk_id = doc.metadata.get("chunk_id")
        pointer = f"{doc.document_id}"
        if page is not None:
            pointer += f", page {page}"
        if chunk_id:
            pointer += f", {chunk_id}"
        excerpt = doc.content.strip().replace("\n", " ")
        if len(excerpt) > 500:
            excerpt = excerpt[:497] + "..."
        lines.append(f"[{index}] ({pointer}, score={doc.score:.3f}) {excerpt}")
    lines.append("")
    lines.append("No claims are made beyond the excerpts above.")
    return "\n".join(lines)


def refusal_message(query: str) -> str:
    return (
        "Insufficient retrieved evidence to answer. "
        f"Refusing to generate an ungrounded response for query: {query}"
    )

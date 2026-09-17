"""Link images, report chunks, RAG, and VLM evidence to canonical building IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from seismic_damage.linking.catalog import build_catalog, normalize_caption
from seismic_damage.schemas.ingestion import TextChunk
from seismic_damage.schemas.linking import BuildingRecord, EvidenceLink
from seismic_damage.schemas.pipeline import RAGResult, VLMObservation, VLMResult


def _image_index(catalog: list[BuildingRecord]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for building in catalog:
        for image_id in building.image_ids:
            mapping[image_id] = building.building_id
    return mapping


def link_image_row(row: dict[str, str], catalog: list[BuildingRecord]) -> EvidenceLink:
    image_id = str(row.get("image_id") or row.get("filename") or "unknown")
    explicit = (row.get("building_id") or "").strip()
    if explicit:
        known = {item.building_id for item in catalog}
        if explicit in known:
            return EvidenceLink(
                evidence_id=image_id,
                evidence_type="image",
                building_id=explicit,
                link_method="explicit_building_id",
                reason="Annotation row already contained a canonical building_id.",
                payload=row,
            )
        return EvidenceLink(
            evidence_id=image_id,
            evidence_type="image",
            building_id=None,
            link_method=None,
            reason="Row building_id is not in the canonical catalog; left unresolved.",
            payload=row,
        )
    mapped = _image_index(catalog).get(image_id)
    if mapped:
        return EvidenceLink(
            evidence_id=image_id,
            evidence_type="image",
            building_id=mapped,
            link_method="caption_group",
            reason="Image belongs to a same-caption group inside one source collection.",
            payload=row,
        )
    return EvidenceLink(
        evidence_id=image_id,
        evidence_type="image",
        building_id=None,
        link_method=None,
        reason="No explicit id and no caption-group membership.",
        payload=row,
    )


def _alias_hits(text: str, catalog: list[BuildingRecord]) -> list[str]:
    haystack = normalize_caption(text)
    hits: list[str] = []
    for building in catalog:
        for alias in building.aliases:
            needle = normalize_caption(alias)
            if needle and needle in haystack:
                hits.append(building.building_id)
                break
        else:
            if building.caption and normalize_caption(building.caption) in haystack:
                hits.append(building.building_id)
    return sorted(set(hits))


def link_chunk(chunk: TextChunk, catalog: list[BuildingRecord]) -> EvidenceLink:
    hits = _alias_hits(chunk.text, catalog)
    if len(hits) == 1:
        return EvidenceLink(
            evidence_id=chunk.chunk_id,
            evidence_type="report_chunk",
            building_id=hits[0],
            link_method="unique_alias",
            reason="Exactly one catalog alias/caption occurred in the chunk.",
            payload=chunk.model_dump(),
        )
    if len(hits) > 1:
        return EvidenceLink(
            evidence_id=chunk.chunk_id,
            evidence_type="report_chunk",
            building_id=None,
            link_method=None,
            reason=f"Ambiguous catalog matches: {hits}. Identity not guessed.",
            payload=chunk.model_dump(),
        )
    return EvidenceLink(
        evidence_id=chunk.chunk_id,
        evidence_type="report_chunk",
        building_id=None,
        link_method=None,
        reason="No unique catalog alias or full caption in chunk text.",
        payload=chunk.model_dump(),
    )


def link_rag_result(
    result: RAGResult,
    catalog: list[BuildingRecord],
    *,
    query_building_id: str | None = None,
) -> list[EvidenceLink]:
    links: list[EvidenceLink] = []
    if query_building_id:
        known = {item.building_id for item in catalog}
        links.append(
            EvidenceLink(
                evidence_id=f"rag_query::{result.query[:80]}",
                evidence_type="rag",
                building_id=query_building_id if query_building_id in known else None,
                link_method="query_building_id" if query_building_id in known else None,
                reason=(
                    "Caller supplied a canonical building_id for this RAG query."
                    if query_building_id in known
                    else "Caller building_id is not in the catalog; left unresolved."
                ),
                payload={"query": result.query, "evidence_sufficient": result.evidence_sufficient},
            )
        )
        return links
    for document in result.documents:
        chunk = TextChunk(
            chunk_id=str(document.metadata.get("chunk_id") or document.document_id),
            document_id=document.document_id,
            source_path=document.source_path or "",
            filename=str(document.metadata.get("filename") or ""),
            page_number=int(document.metadata.get("page_number") or 1),
            chunk_index=int(document.metadata.get("chunk_index") or 0),
            text=document.content,
            n_chars=len(document.content),
            metadata=document.metadata,
        )
        link = link_chunk(chunk, catalog)
        link.evidence_type = "rag"
        links.append(link)
    return links


def link_vlm_observation(
    observation: VLMObservation,
    catalog: list[BuildingRecord],
    *,
    image_id: str | None = None,
) -> EvidenceLink:
    image_ids = _image_index(catalog)
    key = image_id or Path(observation.image_path).stem
    filename = Path(observation.image_path).name
    building_id = image_ids.get(key) or image_ids.get(filename)
    if building_id:
        return EvidenceLink(
            evidence_id=key,
            evidence_type="vlm",
            building_id=building_id,
            link_method="image_id",
            reason="VLM observation mapped through the image's catalog membership.",
            payload={"image_path": observation.image_path, **observation.inferred_attributes},
        )
    return EvidenceLink(
        evidence_id=key,
        evidence_type="vlm",
        building_id=None,
        link_method=None,
        reason="VLM observation has no catalog image_id mapping; identity not guessed from the photo.",
        payload={"image_path": observation.image_path, **observation.inferred_attributes},
    )


def link_vlm_result(result: VLMResult, catalog: list[BuildingRecord]) -> list[EvidenceLink]:
    return [link_vlm_observation(item, catalog) for item in result.observations]


def link_all(
    *,
    image_rows: Iterable[dict[str, str]],
    chunks: Iterable[TextChunk] = (),
    catalog: list[BuildingRecord] | None = None,
) -> tuple[list[BuildingRecord], list[EvidenceLink]]:
    buildings = catalog if catalog is not None else build_catalog(list(image_rows))
    links = [link_image_row(row, buildings) for row in image_rows]
    links.extend(link_chunk(chunk, buildings) for chunk in chunks)
    return buildings, links

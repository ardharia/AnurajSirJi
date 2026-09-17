"""Independent RAG vs VLM vs ground-truth comparison. No prediction fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from seismic_damage.validation.metrics import ALL_PARAMETERS, is_missing, metrics_for

COMPARISONS = (
    ("ground_truth", "rag"),
    ("ground_truth", "vlm"),
    ("rag", "vlm"),
)


def _index_by_building(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        building_id = row.get("building_id")
        if not building_id:
            continue
        indexed[str(building_id)] = row
    return indexed


def _value(row: Mapping[str, Any], parameter: str) -> Any:
    if parameter in row:
        return row.get(parameter)
    nested = row.get("parameters")
    if isinstance(nested, Mapping):
        return nested.get(parameter)
    return None


def compare_sources(
    left_name: str,
    right_name: str,
    left_rows: list[Mapping[str, Any]],
    right_rows: list[Mapping[str, Any]],
    parameters: set[str] | None = None,
) -> dict[str, Any]:
    """Compare two independent sources on intersecting building_ids only."""

    left_index = _index_by_building(left_rows)
    right_index = _index_by_building(right_rows)
    shared = sorted(set(left_index) & set(right_index))
    names = parameters or ALL_PARAMETERS
    per_parameter: dict[str, Any] = {}
    for parameter in sorted(names):
        left_values = [_value(left_index[bid], parameter) for bid in shared]
        right_values = [_value(right_index[bid], parameter) for bid in shared]
        per_parameter[parameter] = metrics_for(parameter, left_values, right_values)
    return {
        "left": left_name,
        "right": right_name,
        "n_left": len(left_index),
        "n_right": len(right_index),
        "n_matched_building_ids": len(shared),
        "matched_building_ids": shared,
        "parameters": per_parameter,
        "fusion": False,
    }


def cross_validate(
    *,
    ground_truth: list[Mapping[str, Any]],
    rag: list[Mapping[str, Any]],
    vlm: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run GT vs RAG, GT vs VLM, and RAG vs VLM. Never ensemble the two models."""

    sources = {
        "ground_truth": list(ground_truth),
        "rag": list(rag),
        "vlm": list(vlm),
    }
    comparisons = [
        compare_sources(left, right, sources[left], sources[right]) for left, right in COMPARISONS
    ]
    return {
        "fusion": False,
        "note": "RAG and VLM remain independent; no averaging, voting, or ensembling.",
        "comparisons": comparisons,
    }


def records_from_vlm_links(links: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse image-level VLM links to building-level records without voting.

    A parameter is kept only when all non-missing values for that building agree.
    Conflicts become missing rather than a majority vote.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for link in links:
        building_id = link.get("building_id")
        if not building_id:
            continue
        payload = link.get("payload") if isinstance(link.get("payload"), Mapping) else link
        grouped[str(building_id)].append(payload)  # type: ignore[arg-type]

    records: list[dict[str, Any]] = []
    for building_id, payloads in grouped.items():
        record: dict[str, Any] = {
            "building_id": building_id,
            "earthquake_event": "Bhuj_2001",
            "evidence_source": "vlm",
        }
        for parameter in ALL_PARAMETERS | {"confidence"}:
            values = [payload.get(parameter) for payload in payloads if not is_missing(payload.get(parameter))]
            unique = []
            for value in values:
                marker = str(value).strip().lower() if not isinstance(value, bool) else value
                if marker not in [str(item).strip().lower() if not isinstance(item, bool) else item for item in unique]:
                    unique.append(value)
            if len(unique) == 1:
                record[parameter] = unique[0]
            # len 0 -> missing; len > 1 -> conflict, leave unset (no vote)
        records.append(record)
    return records

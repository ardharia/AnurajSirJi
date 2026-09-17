"""Canonical building catalog from annotation provenance (no identity guessing)."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from seismic_damage.config.settings import PROJECT_ROOT
from seismic_damage.schemas.linking import BuildingRecord

ANNOTATIONS = PROJECT_ROOT / "data" / "annotations"
OPTIONAL_CATALOG = ANNOTATIONS / "building_catalog.csv"
OPTIONAL_ALIASES = ANNOTATIONS / "building_aliases.csv"

IMAGE_SOURCES = (
    (ANNOTATIONS / "bhuj_2001_iitk_rc_sources.csv", "IITK_NICEE_Bhuj_RC"),
    (ANNOTATIONS / "bhuj_2001_sources.csv", "GEER_Bhuj_2001"),
    (ANNOTATIONS / "bhuj_2001_slideshare_sources.csv", "SlideShare_Bhuj_2001"),
)


def normalize_caption(caption: str | None) -> str:
    text = (caption or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _stable_id(parts: list[str]) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"bldg_{digest}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_image_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path, collection in IMAGE_SOURCES:
        for row in _read_csv(path):
            item = dict(row)
            item.setdefault("source_collection", collection)
            item["_collection"] = item.get("source_collection") or collection
            rows.append(item)
    return rows


def _unique_ngram_aliases(captions_by_building: dict[str, str]) -> dict[str, list[str]]:
    """Aliases that occur in exactly one building caption (n-grams, not fuzzy matches)."""

    index: dict[str, set[str]] = defaultdict(set)
    for building_id, caption in captions_by_building.items():
        words = re.findall(r"[A-Za-z0-9'-]+", caption)
        for n in range(6, 13):
            if len(words) < n:
                continue
            for start in range(0, len(words) - n + 1):
                phrase = " ".join(words[start : start + n])
                if len(phrase) < 40:
                    continue
                index[phrase.lower()].add(building_id)
    aliases: dict[str, list[str]] = defaultdict(list)
    for phrase, owners in index.items():
        if len(owners) != 1:
            continue
        building_id = next(iter(owners))
        aliases[building_id].append(phrase)
    return aliases


def build_catalog(
    image_rows: list[dict[str, str]] | None = None,
    extra_catalog: Path | None = None,
    extra_aliases: Path | None = None,
) -> list[BuildingRecord]:
    """Group images that share an identical caption within a collection.

    Distinct captions are never merged. That would be identity guessing.
    """

    rows = image_rows if image_rows is not None else load_image_rows()
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        collection = row.get("_collection") or row.get("source_collection") or "unknown"
        caption_key = normalize_caption(row.get("caption"))
        if not caption_key:
            caption_key = f"__no_caption__{row.get('image_id') or row.get('filename')}"
        groups[(collection, caption_key)].append(row)

    records: list[BuildingRecord] = []
    captions: dict[str, str] = {}
    for (collection, caption_key), members in groups.items():
        image_ids = [str(item.get("image_id")) for item in members if item.get("image_id")]
        seed = image_ids[0] if image_ids else caption_key
        building_id = _stable_id([collection, caption_key, seed])
        caption = (members[0].get("caption") or "").strip() or None
        if caption:
            captions[building_id] = caption
        records.append(
            BuildingRecord(
                building_id=building_id,
                image_ids=image_ids,
                aliases=[],
                caption=caption,
                source_collection=collection,
                metadata={"n_images": len(members), "caption_key": caption_key},
            )
        )

    by_id = {item.building_id: item for item in records}

    for path in (extra_catalog, OPTIONAL_CATALOG):
        if path is None:
            continue
        for row in _read_csv(path):
            building_id = (row.get("building_id") or "").strip()
            image_id = (row.get("image_id") or "").strip()
            if not building_id:
                continue
            record = by_id.get(building_id)
            if record is None:
                record = BuildingRecord(building_id=building_id)
                by_id[building_id] = record
                records.append(record)
            if image_id and image_id not in record.image_ids:
                record.image_ids.append(image_id)
            alias = (row.get("alias") or "").strip()
            if alias and alias not in record.aliases:
                record.aliases.append(alias)

    auto = _unique_ngram_aliases(captions)
    for building_id, phrases in auto.items():
        record = by_id[building_id]
        for phrase in phrases:
            if phrase not in record.aliases:
                record.aliases.append(phrase)

    for path in (extra_aliases, OPTIONAL_ALIASES):
        if path is None:
            continue
        for row in _read_csv(path):
            building_id = (row.get("building_id") or "").strip()
            alias = (row.get("alias") or "").strip()
            if building_id in by_id and alias and alias not in by_id[building_id].aliases:
                by_id[building_id].aliases.append(alias)

    return records

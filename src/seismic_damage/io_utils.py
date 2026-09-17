"""Small JSON/JSONL helpers for processed research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path | str, payload: Any) -> Path:
    destination = ensure_parent(Path(path))
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return destination


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = ensure_parent(Path(path))
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), default=str))
            handle.write("\n")
    return destination


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

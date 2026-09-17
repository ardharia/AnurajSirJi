"""Building-level frequency analysis, rankings, and Matplotlib figures."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from seismic_damage.io_utils import ensure_parent, write_json, write_jsonl
from seismic_damage.validation.metrics import ALL_PARAMETERS, is_missing, parameter_kind

DEFAULT_OUTPUT_DIR = Path("data/processed/statistics")


def _value(row: Mapping[str, Any], parameter: str) -> Any:
    value = row.get(parameter)
    if is_missing(value) and isinstance(row.get("parameters"), Mapping):
        value = row["parameters"].get(parameter)
    return value


def deduplicate_buildings(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per building_id. Later rows fill only still-missing fields."""

    ordered: dict[str, dict[str, Any]] = {}
    for row in rows:
        building_id = row.get("building_id")
        if not building_id:
            continue
        current = ordered.setdefault(str(building_id), {"building_id": str(building_id)})
        for key, value in dict(row).items():
            if key == "building_id":
                continue
            if is_missing(current.get(key)) and not is_missing(value):
                current[key] = value
    return list(ordered.values())


def frequency_tables(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    buildings = deduplicate_buildings(rows)
    tables: dict[str, Any] = {"n_buildings": len(buildings), "parameters": {}}
    for parameter in sorted(ALL_PARAMETERS):
        values = [_value(row, parameter) for row in buildings]
        present = [value for value in values if not is_missing(value)]
        counter = Counter(str(value) for value in present)
        tables["parameters"][parameter] = {
            "kind": parameter_kind(parameter),
            "n_present": len(present),
            "n_missing": len(values) - len(present),
            "coverage": (len(present) / len(values)) if values else 0.0,
            "frequencies": dict(counter.most_common()),
        }
    return tables


def rank_parameters(
    frequencies: dict[str, Any],
    comparison: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ranks: list[dict[str, Any]] = []
    metrics_block = (comparison or {}).get("parameters") or {}
    for name, stats in frequencies.get("parameters", {}).items():
        metric = metrics_block.get(name) or {}
        performance = metric.get("exact_match")
        if performance is None:
            performance = metric.get("accuracy")
        if performance is None and metric.get("mae") is not None:
            performance = -float(metric["mae"])
        ranks.append(
            {
                "parameter": name,
                "kind": stats.get("kind"),
                "coverage": stats.get("coverage"),
                "n_present": stats.get("n_present"),
                "performance": performance,
                "n_eval": metric.get("n"),
            }
        )
    ranks.sort(key=lambda item: (-(item["coverage"] or 0.0), -(item["performance"] or 0.0)))
    return ranks


def error_distribution(
    left_rows: list[Mapping[str, Any]],
    right_rows: list[Mapping[str, Any]],
    parameter: str = "damage_grade",
) -> dict[str, Any]:
    left = {str(row["building_id"]): row for row in left_rows if row.get("building_id")}
    right = {str(row["building_id"]): row for row in right_rows if row.get("building_id")}
    errors: list[float] = []
    pairs: list[dict[str, Any]] = []
    for building_id in sorted(set(left) & set(right)):
        a = _value(left[building_id], parameter)
        b = _value(right[building_id], parameter)
        if is_missing(a) or is_missing(b):
            continue
        try:
            err = float(a) - float(b)
        except (TypeError, ValueError):
            continue
        errors.append(err)
        pairs.append({"building_id": building_id, "left": a, "right": b, "error": err})
    return {
        "parameter": parameter,
        "n": len(errors),
        "mean_error": (sum(errors) / len(errors)) if errors else None,
        "pairs": pairs,
    }


def render_plots(
    *,
    frequencies: Mapping[str, Any],
    error: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    grade_freq = frequencies.get("parameters", {}).get("damage_grade", {}).get("frequencies", {})
    if grade_freq:
        labels = list(grade_freq.keys())
        values = [grade_freq[label] for label in labels]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color="#4C78A8")
        ax.set_title("Building-level damage grade frequency")
        ax.set_xlabel("damage_grade")
        ax.set_ylabel("count")
        path = output_dir / "damage_grade_frequency.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    coverage = {
        name: stats.get("coverage", 0.0)
        for name, stats in frequencies.get("parameters", {}).items()
    }
    if coverage:
        names = list(coverage.keys())
        vals = [coverage[name] for name in names]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(names, vals, color="#F58518")
        ax.set_xlim(0, 1)
        ax.set_title("Parameter coverage (deduplicated buildings)")
        ax.set_xlabel("fraction non-missing")
        path = output_dir / "parameter_coverage.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    pairs = error.get("pairs") or []
    if pairs:
        errs = [item["error"] for item in pairs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(errs, bins=9, color="#54A24B", edgecolor="white")
        ax.set_title(f"{error.get('parameter')} error distribution")
        ax.set_xlabel("left - right")
        ax.set_ylabel("count")
        path = output_dir / "damage_grade_error_hist.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    return written


def run_statistical_analysis(
    *,
    records: list[Mapping[str, Any]],
    comparison: Mapping[str, Any] | None = None,
    left_rows: list[Mapping[str, Any]] | None = None,
    right_rows: list[Mapping[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    buildings = deduplicate_buildings(records)
    frequencies = frequency_tables(buildings)
    ranks = rank_parameters(frequencies, comparison)
    error = error_distribution(left_rows or [], right_rows or [])
    figures = render_plots(frequencies=frequencies, error=error, output_dir=destination)
    write_json(destination / "frequency_tables.json", frequencies)
    write_json(destination / "parameter_ranks.json", ranks)
    write_json(destination / "error_distribution.json", error)
    write_jsonl(destination / "buildings_deduplicated.jsonl", buildings)
    summary = {
        "n_buildings": len(buildings),
        "n_figures": len(figures),
        "figures": figures,
        "output_dir": str(destination),
    }
    write_json(destination / "summary.json", summary)
    return summary

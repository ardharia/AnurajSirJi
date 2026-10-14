"""Extract NDMA RVS Primer parameters from PDF into authoritative CSV dictionary."""

from __future__ import annotations

import csv
import re
from pathlib import Path
import fitz

from seismic_damage.config.settings import PROJECT_ROOT

PDF_PATH = PROJECT_ROOT / "reference" / "source_documents" / "NDMA_RVS_Primer-43-73.pdf"
OUTPUT_CSV = PROJECT_ROOT / "data" / "reference" / "rvs_parameter_dictionary.csv"

FORMS = [
    ("RC", "Pre", [2, 3]),
    ("RC", "Post", [4, 5]),
    ("BCBM", "Pre", [6, 7]),
    ("BCBM", "Post", [8, 9]),
    ("CM", "Pre", [10, 11]),
    ("CM", "Post", [12, 13]),
    ("RRSM", "Pre", [14, 15, 16]),
    ("RRSM", "Post", [17, 18]),
    ("MH", "Pre", [19, 20]),
    ("MH", "Post", [21, 22]),
    ("DD", "Pre", [23, 24]),
    ("DD", "Post", [25, 26]),
    ("EH", "Pre", [27, 28]),
    ("EH", "Post", [29, 30]),
]


def extract_rvs_dictionary(pdf_path: Path = PDF_PATH, output_csv: Path = OUTPUT_CSV) -> list[dict[str, str]]:
    """Extract typology-specific RVS parameters from NDMA Primer PDF."""
    doc = fitz.open(pdf_path)
    records: list[dict[str, str]] = []

    for typ, phase, pidxs in FORMS:
        current_category = "General"
        form_items: list[dict[str, str | int]] = []

        for p in pidxs:
            words = doc[p].get_text("words")
            lines: dict[int, list[tuple]] = {}
            for w in words:
                line_key = round(w[1] / 6.0) * 6
                lines.setdefault(line_key, []).append(w)

            in_table = False if p == pidxs[0] else True

            for k in sorted(lines.keys()):
                row = lines[k]
                c1_words = [w[4] for w in sorted(row, key=lambda x: x[0]) if w[0] < 165]
                c2_words = [w[4] for w in sorted(row, key=lambda x: x[0]) if 165 <= w[0] < 535]
                c3_words = [w[4] for w in sorted(row, key=lambda x: x[0]) if w[0] >= 535]

                row_text = " ".join(w[4] for w in sorted(row, key=lambda x: x[0]))
                if ("S.No" in row_text or "Parameters" in row_text) and any(t in row_text for t in ["Tag", "Red", "Yellow", "Life Threatening"]):
                    in_table = True
                    continue
                if not in_table:
                    continue
                if any(term in row_text for term in ["None of the above", "FINAL RATING"]) or "Further Actions" in row_text:
                    in_table = False
                    continue

                c1 = " ".join(c1_words).strip()
                c2 = " ".join(c2_words).strip()

                c1_clean = " ".join(c1.split())
                if re.match(r"^\d+\.?\s+[A-Za-z]", c1_clean) or any(cat in c1_clean for cat in ["Siting", "Soil", "Architectural", "Structural", "Construction"]):
                    cat_name = re.sub(r"^\d+\.\s*", "", c1_clean).strip()
                    if cat_name:
                        current_category = cat_name
                elif c1_clean and current_category != "General":
                    if any(w in c1_clean for w in ["Foundation", "Conditions", "Features", "Aspects", "Maintenance", "Details", "details"]):
                        current_category += " " + c1_clean

                current_category = re.sub(r"\s+", " ", current_category).strip()
                m = re.match(r"^(\d+\.\d+)\s*(.*)", c2)
                if m:
                    form_items.append({
                        "typology": typ,
                        "phase": phase,
                        "category": current_category,
                        "item_no": m.group(1),
                        "text": m.group(2).strip(),
                        "page": p + 43,
                    })
                elif c2 and form_items:
                    form_items[-1]["text"] = str(form_items[-1]["text"]) + " " + c2.strip()

        for it in form_items:
            cat = str(it["category"])
            if "Siting" in cat:
                std_cat = "Siting Issues"
            elif "Soil" in cat or "Foundation" in cat:
                std_cat = "Soil & Foundation Conditions"
            elif "Architectur" in cat:
                std_cat = "Architectural Features"
            elif "Structural" in cat:
                std_cat = "Structural Aspects"
            elif "Construction" in cat or "Maintenance" in cat or "Material" in cat:
                std_cat = "Construction and Maintenance Details"
            else:
                std_cat = cat

            item_no = str(it["item_no"])
            param_id = f"{it['typology']}_RVS_{item_no}"
            if it["phase"] == "Post":
                param_id = f"{it['typology']}_POST_{item_no}"

            clean_text = " ".join(str(it["text"]).split())
            clean_text = re.sub(r"\s+(?:Red|Yellow|Green)$", "", clean_text).strip()

            records.append({
                "typology": str(it["typology"]),
                "parameter_id": param_id,
                "parameter_category": std_cat,
                "parameter_text": clean_text,
                "source_document": "NDMA_RVS_Primer-43-73.pdf",
                "source_page": str(it["page"]),
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["typology", "parameter_id", "parameter_category", "parameter_text", "source_document", "source_page"],
        )
        writer.writeheader()
        writer.writerows(records)

    return records


if __name__ == "__main__":
    recs = extract_rvs_dictionary()
    print(f"Extracted {len(recs)} parameters to {OUTPUT_CSV}")
    by_t: dict[str, int] = {}
    for r in recs:
        by_t[r["typology"]] = by_t.get(r["typology"], 0) + 1
    print("Typology counts:", by_t)

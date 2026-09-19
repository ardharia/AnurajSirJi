"""Feature importance and parameter frequency analysis for building vulnerability."""

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from seismic_damage.validation.cross_validation import _index_by_building, _value
from seismic_damage.validation.metrics import ALL_PARAMETERS, is_missing
from seismic_damage.io_utils import ensure_parent


def calculate_feature_importance(
    gt_records: list[Mapping[str, Any]]
) -> pd.DataFrame:
    """Run a Random Forest to determine feature importance for structural damage."""
    
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for feature importance analysis. "
            "Please install it using 'pip install scikit-learn'."
        )

    # 1. Build DataFrame
    gt_idx = _index_by_building(gt_records)
    rows = []
    
    # Define features to include
    exclude = {"damage_grade", "structural_damage_level", "non_structural_damage_level", "retrofitting_action"}
    feature_params = [p for p in ALL_PARAMETERS if p not in exclude]

    for bid, record in gt_idx.items():
        row = {"building_id": bid}
        target = _value(record, "damage_grade")
        row["damage_grade"] = target
        
        for param in feature_params:
            row[param] = _value(record, param)
            
        rows.append(row)

    df = pd.DataFrame(rows)

    # 2. Preprocess
    # Drop rows without a target
    df = df.dropna(subset=["damage_grade"]).copy()
    
    if len(df) == 0:
        return pd.DataFrame()

    y = df["damage_grade"].astype(int)
    X_raw = df[feature_params].copy()

    # Handle missing values
    for col in X_raw.columns:
        if X_raw[col].dtype == object or X_raw[col].dtype == bool:
            X_raw[col] = X_raw[col].fillna("Missing").astype(str)
        else:
            median = X_raw[col].median()
            X_raw[col] = X_raw[col].fillna(median if pd.notna(median) else 0.0)

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X_raw)

    # 3. Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_encoded, y)

    # 4. Aggregate Importances back to base parameters
    raw_importances = clf.feature_importances_
    encoded_cols = X_encoded.columns

    agg_importances = {param: 0.0 for param in feature_params}

    for col, imp in zip(encoded_cols, raw_importances):
        # Determine which base parameter this one-hot column came from
        # pd.get_dummies creates columns like `material_type_RC`
        for param in feature_params:
            if col == param or col.startswith(f"{param}_"):
                agg_importances[param] += imp
                break

    # 5. Format results
    results = []
    for param, imp in agg_importances.items():
        results.append({
            "parameter": param,
            "importance_score": round(imp, 4)
        })

    result_df = pd.DataFrame(results).sort_values("importance_score", ascending=False).reset_index(drop=True)
    return result_df


def run_vulnerability_analysis(
    gt_records: list[Mapping[str, Any]],
    output_dir: Path | str | None = None
) -> dict[str, Any]:
    """Execute the full vulnerability feature analysis."""
    
    try:
        importance_df = calculate_feature_importance(gt_records)
        importance_list = importance_df.to_dict("records")
    except ImportError as e:
        print(f"Warning: {e}")
        importance_df = pd.DataFrame()
        importance_list = []

    report = {
        "n_buildings": len(gt_records),
        "feature_importances": importance_list
    }

    if output_dir and not importance_df.empty:
        base = ensure_parent(Path(output_dir) / ".gitkeep").parent
        base.mkdir(parents=True, exist_ok=True)
        
        importance_df.to_csv(base / "feature_importance.csv", index=False)
        with open(base / "vulnerability_analysis.json", "w") as f:
            json.dump(report, f, indent=2)

    return report

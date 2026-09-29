# Step 15: Robustness / Missing & Poor VLM Data Analysis

## Objective
Evaluate how the complete building assessment framework behaves when visual evidence is systematically missing, subsetted, or degraded.

## Scenarios Analyzed
- **Condition A (100%)**: Full baseline image availability.
- **Condition B (75%)**: Deterministic 75% image subset per building.
- **Condition C (50%)**: Deterministic 50% image subset per building.
- **Condition D (25%)**: Deterministic 25% image subset per building.
- **Condition E (Pixelated)**: Severe nearest-neighbor downscaling/upscaling degradation.
- **Condition F (Low Quality)**: Severe contrast loss and Gaussian blur.
- **Condition G (0%)**: Zero usable visual evidence (VLM explicitly unavailable).
- **Condition H (Random Removal)**: Random removal of one photo for multi-image buildings.
- **Condition I (Primary Removal)**: Removal of the primary (index 0) photo.
- **Condition J (Single Image)**: Retention of only the first photo per building.

## Methodological Rules
1. **Missing ≠ Zero**: When visual evidence is missing or degraded, fields are represented as explicit `None` (`vlm_available = False`).
2. **Modality Comparison**: Compares RAG-only, VLM-only, and Calibrated Multimodal fusion under identical degradation conditions.
3. **Reproducibility**: Governed by deterministic seeding and manifests under `data/processed/robustness/`.

# Step 18: Integrated Seismic Risk Framework

## Formulation
$$\text{Risk} = \text{Hazard Demand} \times \text{Site Amplification} \times \text{Building Vulnerability} \times \text{Exposure Consequence}$$

## Architectural Layers
1. **Regional Hazard**: Peak ground acceleration (PGA), earthquake magnitude ($M_w=7.7$), and seismic zone factor ($Z=0.36$).
2. **Local Site Effects**: Geotechnical soil amplification factors ($1.0$ for rock, $1.36$ for medium soil, $1.67$ for soft soil) and topographic modifiers.
3. **Building Vulnerability**: Step 17 pre-earthquake vulnerability index $V_I \in [0, 1]$.
4. **Building Exposure**: Occupancy consequence weighting (residential, commercial, educational, lifeline/healthcare).

## Outputs
Saved under `data/processed/risk/` including `risk_assessment.csv`, `hazard_summary.csv`, `vulnerability_summary.csv`, and `risk_manifest.json`.

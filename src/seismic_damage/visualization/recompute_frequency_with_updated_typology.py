import pandas as pd
from pathlib import Path
import subprocess

# Paths
OBS_CSV = Path('data/analysis/building_parameter_observations.csv')
SUGGEST_CSV = Path('data/analysis/unknown_typology_keyword_scan.csv')
NEW_FREQ_CSV = Path('data/analysis/typology_parameter_frequency_updated.csv')

# 1️⃣ Load data
obs_df = pd.read_csv(OBS_CSV)
suggest_df = pd.read_csv(SUGGEST_CSV)

# 2️⃣ Build mapping: building_id -> suggested typology (ignore UNKNOWN)
suggest_map = suggest_df.set_index('building_id')['keyword_guess'].to_dict()

# 3️⃣ Update typology for unknown buildings where a guess was made

def apply_guess(row):
    if row['typology'] == 'UNKNOWN':
        guess = suggest_map.get(row['building_id'], 'UNKNOWN')
        if guess != 'UNKNOWN':
            return guess
    return row['typology']

obs_df['typology'] = obs_df.apply(apply_guess, axis=1)

# 4️⃣ Recompute frequency summary (same logic as in frequency_pipeline)
# Count distinct buildings per (event_id, typology, parameter_source, parameter_id, parameter_text)
freq_rows = []
group_cols = ['event_id', 'typology', 'parameter_source', 'parameter_id', 'parameter_text']
for name, group in obs_df.groupby(group_cols):
    event_id, typology, param_src, param_id, param_text = name
    unique_buildings = group['building_id'].nunique()
    freq_rows.append({
        'event_id': event_id,
        'typology': typology,
        'parameter_source': param_src,
        'parameter_id': param_id,
        'parameter': param_text,
        'frequency': unique_buildings,
    })

freq_df = pd.DataFrame(freq_rows)
# Compute total buildings per typology for percentage

total_per_typ = obs_df.drop_duplicates(['building_id', 'typology']).groupby('typology')['building_id'].count()
freq_df['total_buildings'] = freq_df['typology'].map(total_per_typ)
freq_df['frequency_percentage'] = (freq_df['frequency'] / freq_df['total_buildings'] * 100).round(1).astype(str) + '%'

# Save updated frequency CSV
NEW_FREQ_CSV.parent.mkdir(parents=True, exist_ok=True)
freq_df.to_csv(NEW_FREQ_CSV, index=False)
print(f"OK Updated frequency CSV written to {NEW_FREQ_CSV}")

# 5️⃣ Revised building counts per typology
building_counts = obs_df.drop_duplicates(['building_id', 'typology']).groupby('typology')['building_id'].count()
print('\nRevised building counts per typology:')
print(building_counts)

# 6️⃣ Generate plots for each typology (RC, RRSM, BCBM)
plot_script = Path('src/seismic_damage/visualization/plot_parameter_frequency.py')
for typ in ['RC', 'RRSM', 'BCBM']:
    cmd = [
        'python', str(plot_script),
        '--csv', str(NEW_FREQ_CSV),
        '--typology', typ,
        '--out-dir', 'data/analysis/plots',
        '--mode', 'count'
    ]
    print(f"\nRunning plot for typology {typ} ...")
    subprocess.run(cmd, check=True)

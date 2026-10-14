import pandas as pd
from pathlib import Path

CSV_PATH = Path('data/analysis/building_parameter_observations.csv')
df = pd.read_csv(CSV_PATH)

# Filter unknown typology rows
unknown_df = df[df['typology'] == 'UNKNOWN']

# Determine prefix of parameter_id (e.g., RC_, RRSM_, BCBM_)
def prefix(param_id: str) -> str:
    if pd.isna(param_id):
        return ''
    if param_id.startswith('RC_'):
        return 'RC'
    if param_id.startswith('RRSM_'):
        return 'RRSM'
    if param_id.startswith('BCBM_'):
        return 'BCBM'
    return 'OTHER'

unknown_df['prefix'] = unknown_df['parameter_id'].apply(prefix)

# Aggregate per building
suggestions = []
for building_id, group in unknown_df.groupby('building_id'):
    # Count prefixes
    counts = group['prefix'].value_counts()
    if counts.empty:
        suggested = 'UNKNOWN'
    else:
        # Take the most common non-OTHER prefix
        # If top is OTHER, fallback to next
        for pref, _ in counts.items():
            if pref != 'OTHER':
                suggested = pref
                break
        else:
            suggested = 'UNKNOWN'
    suggestions.append({
        'building_id': building_id,
        'total_observed': len(group),
        'suggested_typology': suggested,
        'prefix_counts': counts.to_dict()
    })

# Print a concise report
print('=== Unknown building typology suggestions ===')
for s in suggestions:
    print(f"Building {s['building_id']}: {s['total_observed']} observations, suggested typology -> {s['suggested_typology']}")
    print('  prefix breakdown:', s['prefix_counts'])
    print('---')

# Summary totals
summary = pd.DataFrame(suggestions)
if not summary.empty:
    totals = summary['suggested_typology'].value_counts()
    print('\nOverall suggested typology distribution for UNKNOWN buildings:')
    print(totals)
else:
    print('No UNKNOWN buildings found.')

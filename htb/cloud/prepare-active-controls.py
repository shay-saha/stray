from pathlib import Path
from wildcast.migration_regimes import assemble
cache='data/migration-expanded-dev-v3'
rules='data/migration-active-regimes-v1'
base='runs/migration-v3-goal_route-seed73/best.pt'
assemble(cache,rules,base,'runs/migration-v3-regime-identity',active_fallback='base')
assemble(cache,rules,base,'runs/migration-v3-regime-seasonal')
for method in ['species_sqrt','animal']:
 for seed in [17,42,73]:
  source=f'runs/migration-v3-active-{method}-seed{seed}/best.pt'
  assemble(cache,rules,base,f'runs/migration-v3-regime-active-{method}-seed{seed}',active=source)
print('Assembled eight frozen development routing policies')

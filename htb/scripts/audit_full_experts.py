"""Replay each recorded initialization and compare every selected expert module."""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from wildcast.model import MovementTransformer

torch.set_num_threads(4)
records = []
for path in sorted(Path('runs/cloud').glob('full-*/best-trained.pt')):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('model_family'):
        continue
    settings = ck['training']
    torch.manual_seed(settings['seed'])
    initialized = MovementTransformer(**ck['config'])
    current = initialized.state_dict()
    initial_path = Path(settings['init_checkpoint'])
    prior = torch.load(initial_path, map_location='cpu', weights_only=True)
    reusable = {k: v for k, v in prior['model'].items() if k in current and current[k].shape == v.shape}
    assert prior['manifest']['species'] == ck['manifest']['species']
    initialized.load_state_dict(reusable, strict=False)
    # These experiments use identical species dictionaries, so the trainer's
    # explicit per-species embedding copy is already covered by the state copy.
    modules = {}
    for prefix in ['input.', 'encoder.', 'species.', 'driver_encoder.', 'environment_encoder.',
                   'environment_gate', 'memory_input.', 'memory_encoder.', 'memory_attention.', 'head.']:
        keys = [k for k in ck['model'] if k.startswith(prefix)]
        changed = [k for k in keys if not torch.equal(ck['model'][k], initialized.state_dict()[k])]
        modules[prefix] = {'tensors': len(keys), 'changed_tensors': len(changed)}
    for prefix in ['input.', 'encoder.', 'species.', 'driver_encoder.', 'memory_encoder.', 'memory_attention.', 'head.']:
        assert modules[prefix]['changed_tensors'] > 0, (path, prefix)
    record = {'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'init_checkpoint': str(initial_path), 'init_sha256': hashlib.sha256(initial_path.read_bytes()).hexdigest(),
              'family': settings.get('scope_family'), 'epoch': ck['epoch'],
              'parameters': settings['parameters'], 'trainable_parameters': settings['trainable_parameters'],
              'training_cases': settings['training_cases'], 'validation_cases': settings['validation_cases'],
              'modules': modules, 'environment_gate': float(ck['model'].get('environment_gate', torch.tensor(0.)).item())}
    records.append(record)
Path('reports/full-network-weight-audit.json').write_text(json.dumps({'scope': 'Exact seeded initialization replay against selected post-update checkpoints', 'experts': records}, indent=2)+'\n')
print('Audited complete selected networks:', len(records))

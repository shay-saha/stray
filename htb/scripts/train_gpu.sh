#!/usr/bin/env bash
set -euo pipefail

# Run inside the installed project environment. This script does not provision
# GPUs or incur cloud charges. Increase capacity only after a validation win.
width="${WILDCAST_WIDTH:-128}"
layers="${WILDCAST_LAYERS:-3}"
batch="${WILDCAST_BATCH_SIZE:-256}"
epochs="${WILDCAST_EPOCHS:-25}"
run_root="${WILDCAST_RUN_ROOT:-runs/gpu}"
data_root="${WILDCAST_DATA_ROOT:-data}"

for seed in 42 17 7; do
  python -m wildcast.cli --data "$data_root" train \
    --output "$run_root/seed$seed" --seed "$seed" \
    --width "$width" --layers "$layers" --batch-size "$batch" \
    --epochs "$epochs" --patience 6
done

python -m wildcast.cli --data "$data_root" evaluate \
  --checkpoints "$run_root/seed42/best.pt" "$run_root/seed17/best.pt" "$run_root/seed7/best.pt" \
  --splits val_temporal val_individual --output "$run_root/validation"

# Choose the final configuration from validation before separately running tests.

#!/bin/bash
# Submits all 7 single-fold diagnostic runs on Lawrencium. Run this from the
# repo root (sbatch's working directory becomes the submission directory, and
# the scripts use relative paths like train_transformer/train_transformer.py).
#
# Edit every diagXX_*.slurm's TODOs first (account/partition/qos, conda env
# name, BASE_DIR) - this just fires them all off once that's done.

set -euo pipefail
cd "$(dirname "$0")/../.."

for script in slurm/lawrencium/diag*.slurm; do
    echo "Submitting $script"
    sbatch "$script"
done

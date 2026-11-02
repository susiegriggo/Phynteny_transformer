#!/bin/bash
# Submits all 10 single-fold diagnostic runs on Lawrencium. Self-navigates to
# the repo root before calling sbatch, so it can be run from anywhere - just
# invoke it with a path (./slurm/lawrencium/submit_all.sh or the full path),
# not a bare `submit_all.sh` with no directory component.
#
# Edit every diagXX_*.slurm's --account/--partition/--qos TODOs first (the
# conda env, data path, and PYTHONPATH are already filled in) - this just
# fires them all off once that's done.

set -euo pipefail
cd "$(dirname "$0")/../.."

for script in slurm/lawrencium/diag*.slurm; do
    echo "Submitting $script"
    sbatch "$script"
done

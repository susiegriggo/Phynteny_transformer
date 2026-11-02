# Lawrencium diagnostic sweep

Two stages, both single-fold (fold 1 of the standard 10-fold, `random_state=42`).

**Stage 1 (diag01-07):** isolate the suspected contributors to Reviewer 3's
Figure 4D periodicity comment: the oversized `lambda_penalty` (100 in every
training script vs. the documented default 0.1) and stacking sinusoidal
absolute PE with a learned relative/circular attention bias at the same time.

| # | script | attention | lambda_penalty | PE | tests |
|---|---|---|---|---|---|
| 1 | `diag01_circular_lambda100_pe.slurm` | circular | 100 | on | control, reproduces current behaviour |
| 2 | `diag02_circular_lambda0.1_pe.slurm` | circular | 0.1 | on | lambda fix alone |
| 3 | `diag03_circular_lambda0.1_nope.slurm` | circular | 0.1 | off | lambda fix + drop redundant PE |
| 4 | `diag04_relative_lambda0.1_nope.slurm` | relative (linear, non-circular) | 0.1 | off | no circularity, cleanest config |
| 5 | `diag05_relative_lambda0.1_pe.slurm` | relative (linear, non-circular) | 0.1 | on | no circularity, PE kept as control |
| 6 | `diag06_absolute_lambda0.1_pe.slurm` | absolute (plain self-attention) | 0.1 | on (only option) | does any relative mechanism beat plain absolute PE |
| 7 | `diag07_baseline.slurm` | n/a (MLP) | n/a | n/a | context-free reference |

**Stage 2 (diag08-10):** leave-one-out ablations against the diag02 backbone
(circular attention, lambda_penalty=0.1, PE on, LSTM off) - isolate the
remaining components identified in the "which parts of Phynteny actually
contribute to gene-order modelling" audit.

| # | script | toggle | tests |
|---|---|---|---|
| 8 | `diag08_lstm.slurm` | `--use_lstm` on | does the BiLSTM contribute beyond PE + circular attention |
| 9 | `diag09_no_strand.slurm` | strand zeroed for every gene | does strand/orientation information matter |
| 10 | `diag10_no_length.slurm` | gene length zeroed for every gene | does gene-length information matter |

Protein-language-model (ESM) ablation already exists separately as
`train_order_only.py` / `slurm/train_order_only.slurm` (zeroes protein
embeddings for every gene) - not duplicated here since it predates this sweep.

## Before submitting

All 10 scripts are fully filled in and ready to submit:

- Load `gcc/11.4.0` + `cuda/12.8.0`. **Conda is not activated inside the
  scripts** - activate the `phynteny_transformer` env yourself
  (`conda activate $SCRATCH/big_conda/phynteny_transformer`) before running
  `sbatch`; `sbatch` inherits your shell's environment by default, so the job
  picks up whatever's already active. (An earlier version activated conda
  in-script via `source $SCRATCH/big_conda/etc/profile.d/conda.sh`, but that
  path doesn't exist on this cluster - `conda.sh` lives somewhere else or
  conda needs `conda init` run first.)
- `cd` into the repo checkout
  (`/global/home/users/susiegrigson/GitHubs/Phynteny_transformer`) and export
  `PYTHONPATH` to it explicitly, so `from src...` always resolves this
  checkout regardless of submission directory or conda-env install state.
- Read data from
  `$SCRATCH/Phynteny_data/phynteny_processed_training_files_and_splits_v0.1`
  and write outputs to `$SCRATCH/phynteny_diagnostic_runs/`, where
  `SCRATCH=/global/home/users/susiegrigson/scratch`.
- SBATCH directives (`--account=pc_microbiome`, `--partition=es1`,
  `--qos=es_normal`, `--cpus-per-task=8`, `--gres=gpu:A40:1`, `--time=16:00:00`
  except diag07's lighter 12h) match the confirmed Lawrencium template.
  `--output`/`--error` use `%x-%j.out`/`%x-%j.err`, so each job's log/err
  files are automatically named after its own `--job-name` plus job ID.

Submit individually with `sbatch slurm/lawrencium/diag01_circular_lambda100_pe.slurm`,
or all at once with `slurm/lawrencium/submit_all.sh` (self-navigates to the
repo root, so it can be invoked with a path from anywhere).

## Notes

- `relative` attention (diag04/05) uses the rewritten `RelativePositionAttention`
  (fixed this session - previously computed a non-pairwise, effectively broken
  relative bias). `absolute` attention (diag06) uses `TransformerClassifier`
  (also fixed this session - previously crashed whenever PE was on).
- `diag08`'s BiLSTM now runs *before* the transformer
  (`TransformerClassifierCircularRelativeAttention`, fixed this session) to
  match the manuscript's Methods description and the other two attention
  classes. Previously this class ran transformer-then-LSTM, the opposite
  order, which would have made this comparison architecturally mismatched
  with what the manuscript claims the model does.
- `diag09`/`diag10` use new dedicated scripts (`train_no_strand.py`,
  `train_no_length.py`, mirroring `train_order_only.py`'s pattern) rather than
  the existing `--ignore_strand_gene_length` flag on `EmbeddingDataset`, which
  has stale column indices left over from before the one-hot-width change and
  only zeroes masked tokens' own features rather than every gene's - it's
  currently broken and unused anywhere in this sweep.
- Deliberately deferred: `protein_dropout_rate` (forced context-reliance
  test), per-genome circularity gating, RoPE, and full 10-fold runs - those
  come after this sweep narrows down which config is worth scaling up.

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

Every script has `# TODO` comments marking cluster-specific values that need
filling in once, consistently, across all 10:

- `--account` / `--partition` / `--qos` SBATCH directives (currently copied
  from `example_slurm/example_script.slurm`: `pc_microbiome` / `es2` /
  `es2_normal` - confirm these are right for this allocation).
- The `module load anaconda3` line and `CONDA_ENV` name - set up a conda env
  with this repo's dependencies on Lawrencium first, then point these at it.
- `BASE_DIR` - wherever `data.X.pkl` / `data.y.pkl` end up on Lawrencium
  scratch once the transfer finishes. Outputs are written under
  `$BASE_DIR/diagnostic_runs/`.

Submit individually with `sbatch slurm/lawrencium/diag01_circular_lambda100_pe.slurm`
(run from the repo root, since the scripts use relative paths to the training
scripts), or all at once with `slurm/lawrencium/submit_all.sh`.

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

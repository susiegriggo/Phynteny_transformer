"""
Aggregate the out-of-fold (OOF) validation attention already saved during training
(final_validation_attention.pkl / final_validation_masks.pkl / final_validation_categories.pkl,
one set per fold) into small summary artifacts for comparing attention periodicity across
model types (original / no_sinusoidal / untrained) - Reviewer 3's positional-encoding comment.

Run this ON Setonix, once per model type, directly against the existing training output
directories. It never touches model checkpoints or the multi-terabyte training output beyond
these three small per-fold files, and only writes a tiny summary (KBs) - that's what actually
needs to come back to a local machine, not the raw model outputs.

Usage:
    python3 scripts/analyze_periodicity.py \
        --base_dir /scratch/pawsey1018/grig0076/phynteny_transformer/trained_models \
        --model_type original \
        --folds 1 2 3 4 5 6 7 8 9 10 \
        --out results/periodicity_original.npz
"""
import json
import os
import pickle

import click
import numpy as np
from loguru import logger

PAD_MASK = -2
PAD_CATEGORY = -1


def load_fold(fold_dir):
    with open(os.path.join(fold_dir, "final_validation_attention.pkl"), "rb") as f:
        attn = pickle.load(f)
    with open(os.path.join(fold_dir, "final_validation_masks.pkl"), "rb") as f:
        masks = pickle.load(f)
    with open(os.path.join(fold_dir, "final_validation_categories.pkl"), "rb") as f:
        cats = pickle.load(f)
    return attn, masks, cats


def attention_by_relative_distance(attn_batches, mask_batches, max_lag):
    """
    Mean layer-0, head-averaged attention as a function of signed relative gene
    distance (key position - query position), restricted to real (non-padded) gene
    positions. This is the 1D analogue of the off-diagonal periodicity visible in
    the Figure 4D attention heatmap.

    Returns (lags, mean_profile, pair_counts).
    """
    n_bins = 2 * max_lag + 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    offsets = np.arange(-max_lag, max_lag + 1)

    for attn, mask in zip(attn_batches, mask_batches):
        head_mean = attn.mean(axis=1)  # (batch, seq, seq)
        valid = mask != PAD_MASK  # (batch, seq)
        batch_size = valid.shape[0]

        for b in range(batch_size):
            idx = np.nonzero(valid[b])[0]
            n = idx.size
            if n < 2:
                continue
            sub = head_mean[b][np.ix_(idx, idx)]  # (n, n), real genes only
            rel = np.subtract.outer(np.arange(n), np.arange(n))  # query - key

            in_range = np.abs(rel) <= max_lag
            bins = rel[in_range] + max_lag
            vals = sub[in_range]
            sums += np.bincount(bins, weights=vals, minlength=n_bins)[:n_bins]
            counts += np.bincount(bins, minlength=n_bins)[:n_bins]

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts
    return offsets, profile, counts


def category_agreement_by_relative_distance(mask_batches, cat_batches, max_lag):
    """
    Fraction of gene pairs sharing the same PHROG category, as a function of signed
    relative gene distance, restricted to real gene positions where BOTH genes have
    a known category. This tests Reviewer 3's suggestion: if genes at the periodic
    attention distances don't show elevated functional relationships, that supports
    the periodicity being an architectural artifact rather than learned biology.

    Returns (lags, agreement_rate, pair_counts).
    """
    n_bins = 2 * max_lag + 1
    agree = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    offsets = np.arange(-max_lag, max_lag + 1)

    for mask, cats in zip(mask_batches, cat_batches):
        known = (mask == 1) & (cats != PAD_CATEGORY)  # real gene AND has an annotated category
        batch_size = known.shape[0]

        for b in range(batch_size):
            idx = np.nonzero(known[b])[0]
            n = idx.size
            if n < 2:
                continue
            c = cats[b][idx]
            same = (c[:, None] == c[None, :]).astype(float)
            rel = np.subtract.outer(idx, idx)  # true gene-position distance, not compacted index

            in_range = np.abs(rel) <= max_lag
            bins = rel[in_range] + max_lag
            vals = same[in_range]
            agree += np.bincount(bins, weights=vals, minlength=n_bins)[:n_bins]
            counts += np.bincount(bins, minlength=n_bins)[:n_bins]

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = agree / counts
    return offsets, rate, counts


def quantify_periodicity(offsets, profile, min_period=4, max_period=None):
    """
    Quantify periodicity in the (already background-subtracted-able) relative-distance
    attention profile using autocorrelation over lags 1..max_period.

    Returns a dict: best_period, autocorrelation at that period, and the full
    autocorrelation curve (for plotting), restricted to the positive-lag half of the
    profile (the profile should be roughly symmetric for a well-behaved model).
    """
    center = len(offsets) // 2
    pos_profile = profile[center:]  # lag 0, 1, 2, ... max_lag
    valid = ~np.isnan(pos_profile)
    if valid.sum() < 8:
        return {"best_period": None, "best_autocorr": None, "autocorr_curve": []}

    x = pos_profile.copy()
    x[~valid] = np.nanmean(pos_profile)
    x = x - x.mean()

    if max_period is None:
        max_period = len(x) - 1

    autocorr = []
    lags = list(range(min_period, min(max_period, len(x) - 1) + 1))
    denom = np.sum(x**2)
    for lag in lags:
        num = np.sum(x[:-lag] * x[lag:])
        autocorr.append(float(num / denom) if denom > 0 else 0.0)

    if len(autocorr) == 0:
        return {"best_period": None, "best_autocorr": None, "autocorr_curve": []}

    best_i = int(np.argmax(autocorr))
    return {
        "best_period": lags[best_i],
        "best_autocorr": autocorr[best_i],
        "lags": lags,
        "autocorr_curve": autocorr,
    }


@click.command()
@click.option("--base_dir", required=True, type=click.Path(exists=True), help="Directory containing sinusoidal_comparison_<model_type>_fold_<N> directories.")
@click.option("--model_type", required=True, type=click.Choice(["original", "no_sinusoidal", "untrained"]), help="Which model type to aggregate.")
@click.option("--folds", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated list of fold indices to include.")
@click.option("--max_lag", default=40, type=int, help="Maximum relative gene distance to consider (should comfortably cover a few multiples of the reported ~15-gene period).")
@click.option("--out", required=True, type=click.Path(), help="Output .npz path for the summary (small - safe to copy back locally).")
def main(base_dir, model_type, folds, max_lag, out):
    fold_list = [int(f) for f in folds.split(",")]

    all_attn, all_masks, all_cats = [], [], []
    for fold in fold_list:
        fold_dir = os.path.join(base_dir, f"sinusoidal_comparison_{model_type}_fold_{fold}", f"fold_{fold}")
        if not os.path.exists(fold_dir):
            logger.warning(f"Skipping missing fold directory: {fold_dir}")
            continue
        logger.info(f"Loading fold {fold} from {fold_dir}")
        attn, masks, cats = load_fold(fold_dir)
        all_attn.extend(attn)
        all_masks.extend(masks)
        all_cats.extend(cats)

    logger.info(f"Loaded {len(all_attn)} validation batches across {len(fold_list)} folds for model_type={model_type}")

    offsets, attn_profile, attn_counts = attention_by_relative_distance(all_attn, all_masks, max_lag)
    logger.info("Computed attention-by-relative-distance profile")

    cat_offsets, agreement_profile, agreement_counts = category_agreement_by_relative_distance(all_masks, all_cats, max_lag)
    logger.info("Computed category-agreement-by-relative-distance profile")

    periodicity = quantify_periodicity(offsets, attn_profile)
    logger.info(f"Periodicity summary: {periodicity.get('best_period')=} {periodicity.get('best_autocorr')=}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(
        out,
        model_type=model_type,
        folds=np.array(fold_list),
        offsets=offsets,
        attn_profile=attn_profile,
        attn_counts=attn_counts,
        cat_offsets=cat_offsets,
        agreement_profile=agreement_profile,
        agreement_counts=agreement_counts,
        periodicity_lags=np.array(periodicity.get("lags", [])),
        periodicity_autocorr=np.array(periodicity.get("autocorr_curve", [])),
        periodicity_best_period=periodicity.get("best_period") if periodicity.get("best_period") is not None else -1,
        periodicity_best_autocorr=periodicity.get("best_autocorr") if periodicity.get("best_autocorr") is not None else np.nan,
    )
    logger.info(f"Saved summary to {out}")

    with open(str(out) + ".json", "w") as f:
        json.dump(
            {
                "model_type": model_type,
                "folds": fold_list,
                "n_batches": len(all_attn),
                "best_period": periodicity.get("best_period"),
                "best_autocorr": periodicity.get("best_autocorr"),
            },
            f,
            indent=2,
        )
    logger.info("Done")


if __name__ == "__main__":
    main()

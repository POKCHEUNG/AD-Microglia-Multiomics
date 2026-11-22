"""Depth-matched neighbourhood composition around two states of a centre cell type.

For centre cells (e.g. microglia) split into a focal state (e.g. disease-associated) and
the rest, compare the fraction of target cells (e.g. vulnerable neurons) among all
neighbours of interest within ``radius``. The null permutes the focal-state label among
centre cells **within (section, depth bin)**, which preserves laminar structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def neighbour_counts(xy: np.ndarray, centres: np.ndarray, masks: dict[str, np.ndarray], radius: float) -> dict[str, np.ndarray]:
    """For each centre index, number of cells in each ``masks`` category within ``radius``
    (the centre itself is excluded if it belongs to the category)."""
    out = {}
    for name, m in masks.items():
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            out[name] = np.zeros(len(centres), int)
            continue
        tree = cKDTree(xy[idx])
        n = np.asarray(tree.query_ball_point(xy[centres], r=radius, return_length=True))
        out[name] = n - m[centres].astype(int)
    return out


def _stat(focal: np.ndarray, target: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """(log2 ratio of target fraction among neighbours, log2 ratio of mean neighbour count)
    for focal vs non-focal centres."""
    tot = target + other
    a_t, a_n = target[focal].sum(), tot[focal].sum()
    b_t, b_n = target[~focal].sum(), tot[~focal].sum()
    frac = np.log2(((a_t + 0.5) / (a_n + 1)) / ((b_t + 0.5) / (b_n + 1)))
    dens = np.log2((tot[focal].mean() + 0.1) / (tot[~focal].mean() + 0.1))
    return frac, dens


def section_test(xy: np.ndarray, depth_bin: np.ndarray, centre: np.ndarray, focal: np.ndarray, target: np.ndarray,
                 other: np.ndarray, radius: float, n_perm: int = 1000, seed: int = 0, min_focal: int = 10) -> dict | None:
    """One section. ``centre``/``focal``/``target``/``other`` are boolean masks over cells;
    ``focal`` must be a subset of ``centre``. Returns observed statistics, permutation-null
    means/SDs and the excess (observed - null mean), or None if too few focal centres."""
    c = np.flatnonzero(centre)
    f = focal[c]
    if f.sum() < min_focal or (~f).sum() < min_focal:
        return None
    cnt = neighbour_counts(xy, c, {"target": target, "other": other}, radius)
    obs_frac, obs_dens = _stat(f, cnt["target"], cnt["other"])
    rng = np.random.default_rng(seed)
    bins = depth_bin[c]
    groups = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    null = np.empty((n_perm, 2))
    for i in range(n_perm):
        fp = f.copy()
        for g in groups:
            fp[g] = rng.permutation(f[g])
        null[i] = _stat(fp, cnt["target"], cnt["other"])
    return {"n_focal": int(f.sum()), "n_centre": len(c),
            "frac_obs": obs_frac, "frac_null": null[:, 0].mean(), "frac_null_sd": null[:, 0].std(),
            "dens_obs": obs_dens, "dens_null": null[:, 1].mean(), "dens_null_sd": null[:, 1].std(),
            "frac_excess": obs_frac - null[:, 0].mean(), "dens_excess": obs_dens - null[:, 1].mean()}


def depth_bins(depth: pd.Series, section: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Quantile bins of cortical depth within each section; missing depth gets its own bin."""
    out = np.full(len(depth), -1)
    for _, idx in depth.groupby(section, observed=True).groups.items():
        pos = depth.index.get_indexer(idx)
        d = depth.iloc[pos]
        ok = d.notna().to_numpy()
        if ok.sum() >= n_bins:
            out[pos[ok]] = pd.qcut(d[ok].rank(method="first"), n_bins, labels=False).to_numpy()
    return out

"""Cell-level test for a minority, coordinated DDR state within neuronal supertypes.

A real damage-response state should (i) co-elevate several repair / p53 programs in the
same nucleus more than expression-matched random gene sets do, and (ii) be visible in
chromatin of the same nucleus (paired ATAC), which RNA sampling noise would not be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def matched_random_programs(scorer_mean: np.ndarray, var_names: pd.Index, programs: dict[str, list[str]],
                            n: int, n_bins: int = 25, seed: int = 0) -> list[dict[str, list[str]]]:
    """``n`` replicates; each replaces every program by a random gene set of the same size,
    matched gene-by-gene on mean expression bin, without overlap with the real programs."""
    rng = np.random.default_rng(seed)
    bins = pd.qcut(pd.Series(scorer_mean).rank(method="first"), n_bins, labels=False).to_numpy()
    upper = {g.upper(): i for i, g in enumerate(var_names)}
    real = {upper[g.upper()] for gs in programs.values() for g in gs if g.upper() in upper}
    pool = np.setdiff1d(np.flatnonzero(scorer_mean > 0), list(real))
    by_bin = {b: pool[bins[pool] == b] for b in np.unique(bins[pool])}
    reps = []
    for _ in range(n):
        rep = {}
        for name, gs in programs.items():
            idx = [upper[g.upper()] for g in gs if g.upper() in upper]
            rep[name] = [var_names[rng.choice(by_bin[bins[i]])] for i in idx if len(by_bin.get(bins[i], []))]
        reps.append(rep)
    return reps


def donor_centred_z(obs: pd.DataFrame, cols: list[str], groupby: str = "supertype", donor: str = "donor") -> pd.DataFrame:
    """Centre each score on its (donor, group) mean and scale by the group's pooled within-donor SD."""
    g = obs.groupby([donor, groupby], observed=True)
    c = obs[cols] - g[cols].transform("mean")
    sd = c.groupby(obs[groupby], observed=True).transform("std")
    return c / sd.where(sd > 0)


def coherence(z: pd.DataFrame, groups: pd.Series, cols: list[str], min_cells: int = 200) -> pd.Series:
    """Mean pairwise within-cell correlation between program scores, per group."""
    out = {}
    for g, d in z.groupby(groups, observed=True):
        if len(d) < min_cells:
            continue
        r = np.corrcoef(d[cols].to_numpy().T)
        out[g] = r[np.triu_indices(len(cols), 1)].mean()
    return pd.Series(out)


def composite(z: pd.DataFrame, cols: list[str]) -> pd.Series:
    return z[cols].mean(1)


def tail_fraction(score: pd.Series, groups: pd.Series, thr: float = 2.5, min_cells: int = 200) -> pd.Series:
    """Fraction of cells whose re-standardised composite exceeds ``thr`` SD, per group."""
    out = {}
    for g, s in score.groupby(groups, observed=True):
        if len(s) < min_cells:
            continue
        zs = (s - s.mean()) / s.std()
        out[g] = float((zs > thr).mean())
    return pd.Series(out)


def top_fraction_labels(score: pd.Series, obs: pd.DataFrame, frac: float = 0.05, groupby: str = "supertype",
                        donor: str = "donor", min_cells: int = 40) -> pd.Series:
    """'high' for the top ``frac`` of cells within each (donor, group), 'rest' otherwise;
    NaN for (donor, group) with fewer than ``min_cells`` nuclei."""
    lab = pd.Series(np.nan, index=score.index, dtype=object)
    for _, idx in obs.groupby([donor, groupby], observed=True).groups.items():
        if len(idx) < min_cells:
            continue
        s = score.loc[idx]
        lab.loc[idx] = np.where(s >= s.quantile(1 - frac), "high", "rest")
    return lab

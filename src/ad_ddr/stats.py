"""Donor-level association of program scores with AD progression (CPS).

With ~12 donors, cells are not replicates: every test is done on donor pseudobulks,
and p-values come from permuting CPS across donors (exact enough at n_perm=10000).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ddrmulti.stats import bh


def _resid(y: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    if Z is None:
        return y - y.mean()
    Z1 = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ beta


def _covar_matrix(df: pd.DataFrame, covariates: list[str]) -> np.ndarray | None:
    if not covariates:
        return None
    return pd.get_dummies(df[covariates], drop_first=True, dtype=float).to_numpy()


def cps_association(pb: pd.DataFrame, cols: list[str], groupby: str = "subclass", x: str = "cps",
                    covariates: list[str] | None = None, min_donors: int = 6, n_perm: int = 10_000,
                    seed: int = 0) -> pd.DataFrame:
    """For each (group, score): partial Pearson r and slope of score on ``x``, adjusting for
    ``covariates``; permutation p (two-sided) by shuffling residualised ``x`` across donors.
    BH q-values are computed within each score column family (across groups)."""
    rng = np.random.default_rng(seed)
    covariates = covariates or []
    rows = []
    for grp, d in pb.groupby(groupby, observed=True):
        if len(d) < min_donors:
            continue
        for c in cols:
            dc = d[d[c].notna()]
            if len(dc) < min_donors:
                continue
            Z = _covar_matrix(dc, covariates)
            xr = _resid(dc[x].to_numpy(float), Z)
            perms = np.stack([rng.permutation(xr) for _ in range(n_perm)])
            yr = _resid(dc[c].to_numpy(float), Z)
            if yr.std() == 0 or xr.std() == 0:
                continue
            r = np.corrcoef(xr, yr)[0, 1]
            slope = (xr @ yr) / (xr @ xr)
            rp = (perms - perms.mean(1, keepdims=True)) @ (yr - yr.mean())
            rp /= np.linalg.norm(perms - perms.mean(1, keepdims=True), axis=1) * np.linalg.norm(yr - yr.mean())
            p = (np.sum(np.abs(rp) >= abs(r) - 1e-12) + 1) / (n_perm + 1)
            rows.append({groupby: grp, "score": c, "r": r, "slope": slope, "p_perm": p, "n_donors": len(dc)})
    res = pd.DataFrame(rows, columns=[groupby, "score", "r", "slope", "p_perm", "n_donors"])
    if res.empty:
        return res.assign(q=pd.Series(dtype=float))
    res["q"] = res.groupby("score")["p_perm"].transform(lambda p: bh(p.to_numpy()))
    return res.sort_values("p_perm").reset_index(drop=True)


def rna_atac_concordance(pb_rna: pd.DataFrame, pb_atac: pd.DataFrame, programs: list[str],
                         groupby: str = "subclass", donor: str = "donor", min_donors: int = 6) -> pd.DataFrame:
    """Across donors within each group: correlation of RNA program score with promoter
    accessibility of the same program."""
    m = pb_rna.merge(pb_atac, on=[groupby, donor], suffixes=("", "_atac"))
    rows = []
    for grp, d in m.groupby(groupby, observed=True):
        for p in programs:
            a, b = f"rna_{p}", f"atac_{p}"
            if a not in d or b not in d:
                continue
            dd = d[[a, b]].dropna()
            if len(dd) >= min_donors:
                rows.append({groupby: grp, "program": p, "r": np.corrcoef(dd[a], dd[b])[0, 1], "n_donors": len(dd)})
    return pd.DataFrame(rows, columns=[groupby, "program", "r", "n_donors"])

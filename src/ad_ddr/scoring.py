"""Program scores that can be computed one donor at a time.

RNA: ``ProgramScorer`` fixes each program's genes and expression-matched control genes
once (Seurat / ``sc.tl.score_genes`` recipe), so scores are comparable across donors
scored separately. ATAC: counts are summed per (group, donor) as donors stream in,
then scored as promoter-set enrichment over matched background peaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ddrmulti import atac as datac

from .signatures import all_programs


def cell_groups(obs: pd.DataFrame) -> pd.Series:
    """Subclass, except the SEA-AD 'Immune' subclass is reduced to microglia/PVM
    (``Micro-PVM_*`` supertypes); lymphocytes are dropped (NaN)."""
    g = obs["subclass"].astype(object).copy()
    imm = g == "Immune"
    micro = obs["supertype"].astype(str).str.startswith("Micro-PVM")
    g[imm & micro] = "Microglia-PVM"
    g[imm & ~micro] = np.nan
    return g


class ProgramScorer:
    """Mean ln(UP10K+1) of program genes minus mean of expression-matched control genes."""

    def __init__(self, programs: dict | None = None, n_bins: int = 25, ctrl_size: int = 50,
                 min_genes: int = 3, seed: int = 0):
        self.programs = programs or all_programs()
        self.n_bins, self.ctrl_size, self.min_genes, self.seed = n_bins, ctrl_size, min_genes, seed
        self.sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, ref: AnnData) -> "ProgramScorer":
        rng = np.random.default_rng(self.seed)
        mean = np.asarray(sp.csr_matrix(ref.X).mean(0)).ravel()
        self.mean_ = mean
        bins = pd.qcut(pd.Series(mean).rank(method="first"), self.n_bins, labels=False).to_numpy()
        upper = {g.upper(): i for i, g in enumerate(ref.var_names)}
        self.var_names = ref.var_names.copy()
        for name, genes in self.programs.items():
            idx = np.array(sorted({upper[g.upper()] for g in genes if g.upper() in upper}))
            if len(idx) < self.min_genes:
                continue
            pool = np.setdiff1d(np.arange(len(mean)), idx)
            ctrl = np.concatenate([rng.choice(pool[bins[pool] == b], self.ctrl_size, replace=False)
                                   for b in bins[idx]])
            self.sets[name] = (idx, np.unique(ctrl))
        return self

    def score(self, adata: AnnData, prefix: str = "rna_") -> pd.DataFrame:
        if not adata.var_names.equals(self.var_names):
            adata = adata[:, self.var_names]
        X = sp.csc_matrix(adata.X)                      # column slicing is cheap in CSC
        out = {}
        for name, (idx, ctrl) in self.sets.items():
            out[f"{prefix}{name}"] = (np.asarray(X[:, idx].mean(1)).ravel()
                                      - np.asarray(X[:, ctrl].mean(1)).ravel())
        return pd.DataFrame(out, index=adata.obs_names)


def depth_adjust(obs: pd.DataFrame, cols: list[str], groupby: str = "group", donor: str = "donor",
                 depth: str = "log_umi", batch: str | None = "method") -> pd.DataFrame:
    """Remove per-cell sequencing-depth (and chemistry) effects from scores.

    Within each cell group the slope on depth is estimated from **within-donor**
    variation only (variables centred per donor), so the estimate is not driven by
    between-donor differences, which carry the CPS signal. The correction is then
    applied to absolute depth, moving every cell to the group's mean depth /
    reference chemistry, which also removes donor-level differences in sequencing depth.
    """
    obs = obs.copy()
    for _, d in obs.groupby(groupby, observed=True):
        Z = d[[depth]].astype(float)
        if batch and d[batch].nunique() > 1:
            Z = Z.join(pd.get_dummies(d[batch], drop_first=True, dtype=float))
        Zc = Z - Z.groupby(d[donor]).transform("mean")
        Y = d[cols].astype(float)
        Yc = Y - Y.groupby(d[donor]).transform("mean")
        beta, *_ = np.linalg.lstsq(Zc.to_numpy(), Yc.to_numpy(), rcond=None)
        obs.loc[d.index, cols] = Y.to_numpy() - (Z - Z.mean()).to_numpy() @ beta
    return obs


def pseudobulk(obs: pd.DataFrame, cols: list[str], groupby: str = "group", donor: str = "donor",
               min_cells: int = 20, covariates: tuple[str, ...] = ("cps", "age", "sex", "pmi", "adnc")) -> pd.DataFrame:
    """Mean score per (group, donor); drops groups with < ``min_cells`` nuclei in a donor."""
    g = obs.groupby([groupby, donor], observed=True)
    pb = g[cols].mean()
    pb["n_cells"] = g.size()
    cov = obs.groupby(donor, observed=True)[list(covariates)].first()
    pb = pb[pb["n_cells"] >= min_cells].reset_index().merge(cov, left_on=donor, right_index=True)
    return pb


def promoter_regions(genes: pd.DataFrame, names: list[str], flank: int = 1000) -> pd.DataFrame:
    g = genes.loc[genes.index.intersection(names)]
    return pd.DataFrame({"chrom": g["chrom"], "start": np.maximum(g["tss"] - flank, 0), "end": g["tss"] + flank})


class AtacAccumulator:
    """Sum snATAC counts per (group, donor) across donors processed one at a time."""

    def __init__(self):
        self.counts: dict[tuple[str, str], np.ndarray] = {}
        self.n_cells: dict[tuple[str, str], int] = {}
        self.var: pd.DataFrame | None = None

    def add(self, atac: AnnData, groups: pd.Series, donor: str):
        if self.var is None:
            self.var = atac.var[["chrom", "start", "end"]].copy()
        elif not atac.var_names.equals(self.var.index):
            raise ValueError("peak sets differ between donors")
        X = sp.csr_matrix(atac.X)
        for g, idx in groups.groupby(groups, observed=True).indices.items():
            self.counts[(g, donor)] = np.asarray(X[idx].sum(0)).ravel()
            self.n_cells[(g, donor)] = len(idx)

    def score(self, genes: pd.DataFrame, programs: dict | None = None, flank: int = 1000, n_bg: int = 100,
              n_bins: int = 20, min_cells: int = 20, seed: int = 0, groupby: str = "group",
              donor: str = "donor") -> pd.DataFrame:
        """log2(counts in program promoter peaks / mean counts in accessibility-matched
        background peak sets) per (group, donor)."""
        programs = programs or all_programs()
        rng = np.random.default_rng(seed)
        keys = [k for k in self.counts if self.n_cells[k] >= min_cells]
        C = np.vstack([self.counts[k] for k in keys])
        frac = C.sum(0) / C.sum()
        strata = pd.qcut(np.log(frac + 1e-12), n_bins, labels=False, duplicates="drop")
        by_bin = {b: np.flatnonzero(strata == b) for b in np.unique(strata)}
        holder = AnnData(sp.csr_matrix((1, len(self.var))), var=self.var)

        out = pd.DataFrame(keys, columns=[groupby, donor])
        for name, g in programs.items():
            peaks = np.flatnonzero(datac.peakset_mask(holder, promoter_regions(genes, g, flank)))
            if len(peaks) < 3:
                continue
            bg = np.stack([np.concatenate([rng.choice(by_bin[b], (strata[peaks] == b).sum())
                                           for b in np.unique(strata[peaks])]) for _ in range(n_bg)])
            obs_counts = C[:, peaks].sum(1)
            bg_counts = C[:, bg].sum(2).mean(1)
            out[f"atac_{name}"] = np.log2((obs_counts + 1) / (bg_counts + 1))
        out["n_cells_atac"] = [self.n_cells[k] for k in keys]
        return out

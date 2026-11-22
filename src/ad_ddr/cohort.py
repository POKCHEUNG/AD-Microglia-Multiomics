"""Full-cohort analysis on SEA-AD's released pseudobulk objects (all donors, 10 regions).

Each released object holds summed UMI counts per (region, supertype, library). We sum
those to (region, cell group, donor), keep genes expressed in the group, and score
programs as the mean z-scored log2 CPM of program genes across donors.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

EXC, INH = "Neuronal: Glutamatergic", "Neuronal: GABAergic"
DONOR_COLS = {
    "Age at Death": "age",
    "Sex": "sex",
    "PMI": "pmi",
    "RIN": "rin",
    "Overall AD neuropathological Change": "adnc",
    "Braak": "braak",
    "Cognitive Status": "cognition",
    "APOE Genotype": "apoe",
    "Last CASI Score": "casi",
    "Interval from last CASI in months": "casi_interval",
    "Severely Affected Donor": "severe",
}


def _group(obs: pd.DataFrame) -> pd.Series:
    g = obs["Subclass"].astype(str).copy()
    imm = g == "Immune"
    micro = obs["Supertype"].astype(str).str.startswith("Micro-PVM")
    g[imm & micro] = "Microglia-PVM"
    g[imm & ~micro] = ""
    return g


def aggregate(paths: list[Path], regions: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray, pd.Index, pd.DataFrame]:
    """Sum released pseudobulk counts to (region, group, donor) and to (region, class, donor)
    for the two neuronal classes. Returns (index table, counts matrix, genes, donor metadata)."""
    keys, mats, donor_meta, genes = [], [], [], None
    for p in paths:
        a = ad.read_h5ad(p)
        if regions:
            a = a[a.obs["Brain Region"].isin(regions)]
        if genes is None:
            genes = a.var_names.copy()
        elif not a.var_names.equals(genes):
            a = a[:, genes]
        obs = a.obs
        donor_meta.append(obs.drop_duplicates("Donor ID").set_index("Donor ID")[list(DONOR_COLS)])
        X = sp.csr_matrix(a.X)
        for level, labels in (("group", _group(obs)), ("class", obs["Class"].astype(str))):
            if level == "class" and not labels.isin([EXC, INH]).any():
                continue
            key = pd.DataFrame({"region": obs["Brain Region"].astype(str).to_numpy(),
                                "group": labels.to_numpy(), "donor": obs["Donor ID"].astype(str).to_numpy(),
                                "n_nuclei": obs["Number of nuclei"].to_numpy(float)})
            key["multiome_nuclei"] = key["n_nuclei"] * (obs["method"].astype(str) == "10xMulti").to_numpy()
            key = key[key["group"] != ""]
            key["level"] = level
            cat = pd.Categorical(key["region"] + "|" + key["group"] + "|" + key["donor"])
            A = sp.csr_matrix((np.ones(len(key)), (cat.codes, key.index.to_numpy())), shape=(len(cat.categories), a.n_obs))
            mats.append(sp.csr_matrix(A @ X))
            k = key.assign(cat=cat).groupby("cat", observed=True).agg(
                region=("region", "first"), group=("group", "first"), donor=("donor", "first"),
                level=("level", "first"), n_nuclei=("n_nuclei", "sum"), multiome_nuclei=("multiome_nuclei", "sum"))
            keys.append(k.reindex(cat.categories).reset_index(drop=True))
        print(f"  {Path(p).stem}: {a.n_obs:,} pseudobulk rows", flush=True)
        del a, X
    idx = pd.concat(keys, ignore_index=True)
    C = sp.vstack(mats).tocsr()
    # class-level rows were added once per subclass file: sum duplicates
    cat = pd.Categorical(idx["level"] + "|" + idx["region"] + "|" + idx["group"] + "|" + idx["donor"])
    A = sp.csr_matrix((np.ones(len(idx)), (cat.codes, np.arange(len(idx)))))
    C = np.asarray((A @ C).todense())
    idx = idx.assign(cat=cat).groupby("cat", observed=True).agg(
        level=("level", "first"), region=("region", "first"), group=("group", "first"), donor=("donor", "first"),
        n_nuclei=("n_nuclei", "sum"), multiome_nuclei=("multiome_nuclei", "sum")).reindex(cat.categories).reset_index(drop=True)
    meta = pd.concat(donor_meta)
    meta = meta[~meta.index.duplicated()].rename(columns=DONOR_COLS)
    return idx, C, genes, meta


def score_programs(idx: pd.DataFrame, C: np.ndarray, genes: pd.Index, programs: dict, min_cpm: float = 1.0,
                   min_frac: float = 0.5, min_nuclei: int = 20, min_genes: int = 3) -> pd.DataFrame:
    """Per (level, region, group): log2 CPM, keep genes with CPM >= ``min_cpm`` in >= ``min_frac``
    of donors, z-score each gene across donors, average over program genes."""
    upper = pd.Series(np.arange(len(genes)), index=genes.str.upper())
    upper = upper[~upper.index.duplicated()]
    out = []
    keep_rows = idx["n_nuclei"] >= min_nuclei
    for (level, region, group), d in idx[keep_rows].groupby(["level", "region", "group"], observed=True):
        if len(d) < 6:
            continue
        M = C[d.index.to_numpy()]
        cpm = M / M.sum(1, keepdims=True) * 1e6
        expressed = (cpm >= min_cpm).mean(0) >= min_frac
        L = np.log2(cpm + 1)
        sd = L.std(0)
        Z = (L - L.mean(0)) / np.where(sd > 0, sd, 1)
        row = d[["level", "region", "group", "donor", "n_nuclei", "multiome_nuclei"]].copy()
        for name, gs in programs.items():
            gi = upper.reindex([g.upper() for g in gs]).dropna().astype(int).to_numpy()
            gi = gi[expressed[gi]]
            row[f"rna_{name}"] = Z[:, gi].mean(1) if len(gi) >= min_genes else np.nan
            row[f"ngenes_{name}"] = len(gi)
        out.append(row)
    res = pd.concat(out, ignore_index=True)
    res["log_nuclei"] = np.log10(res["n_nuclei"])
    res["frac_multiome"] = res["multiome_nuclei"] / res["n_nuclei"]
    return res


def attach_donor_info(scores: pd.DataFrame, meta: pd.DataFrame, cps_csv: str | Path) -> pd.DataFrame:
    cps = pd.read_csv(cps_csv)[["Donor ID", "Brain Region", "CPS_Local", "CPS_Global"]]
    cps.columns = ["donor", "region", "cps", "cps_global"]
    out = scores.merge(cps, on=["donor", "region"], how="left").merge(meta, left_on="donor", right_index=True, how="left")
    out["severe"] = out["severe"].astype(str).eq("Y")
    out["dementia"] = out["cognition"].astype(str).eq("Dementia").astype(float)
    out["apoe4"] = out["apoe"].astype(str).str.count("4").astype(float)
    return out

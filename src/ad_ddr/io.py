"""Load per-donor SEA-AD MTG RNA + ATAC objects."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from ddrmulti.genome import parse_peak_names

CPS_TABLE = Path(__file__).resolve().parents[2] / "data/meta/Global_and_Local_CPS.csv"
OBS_KEEP = {
    "Donor ID": "donor",
    "Class": "class",
    "Subclass": "subclass",
    "Supertype": "supertype",
    "method": "method",
    "Number of UMIs": "n_umi",
    "Genes detected": "n_genes",
    "Overall AD neuropathological Change": "adnc",
    "Braak": "braak",
    "Age at Death": "age",
    "Sex": "sex",
    "PMI": "pmi",
    "RIN": "rin",
    "APOE Genotype": "apoe",
    "Cognitive Status": "cognition",
}
CATEGORICAL = ("donor", "class", "subclass", "supertype", "method", "adnc", "braak", "sex", "apoe", "cognition")


def mtg_cps(path: str | Path = CPS_TABLE) -> pd.Series:
    """2026 multiregion release local CPS for MTG (supersedes the CPS stored in the
    2024 MTG ATAC objects)."""
    t = pd.read_csv(path)
    return t[t["Brain Region"] == "MTG"].set_index("Donor ID")["CPS_Local"]


def load_donor(rna_path: str | Path, atac_path: str | Path, cps: pd.Series | None = None) -> tuple[AnnData, AnnData]:
    """Return (rna, atac) for one donor.

    rna: **all** final nuclei (10x v3.1 singleome + multiome); ``X`` = ln(UP10K+1).
    atac: multiome nuclei that also passed RNA QC, annotated with the RNA cell labels;
    ``X`` = raw fragment counts. ``obs['cps']`` = MTG local CPS from ``mtg_cps()``.
    """
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    cps = (mtg_cps() if cps is None else cps)[str(rna.obs["Donor ID"].iloc[0])]

    obs = rna.obs[list(OBS_KEEP)].rename(columns=OBS_KEEP)
    for c in CATEGORICAL:
        obs[c] = obs[c].astype(str)
    obs["cps"] = float(cps)
    obs["log_umi"] = np.log10(obs["n_umi"].astype(float))
    rna = AnnData(rna.X, obs=obs, var=rna.var[[]])

    atac = atac[atac.obs["method"] == "10xMulti"]
    paired = atac.obs_names.intersection(rna.obs_names[rna.obs["method"] == "10xMulti"])
    atac = AnnData(atac[paired].X, obs=obs.loc[paired].copy(), var=parse_peak_names(atac.var_names))
    return rna, atac


def donor_paths(raw_dir: str | Path, donor: str) -> tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    return raw_dir / "rna" / f"{donor}.h5ad", raw_dir / "atac" / f"{donor}.h5ad"


def selected_donors(meta_csv: str | Path) -> list[str]:
    return pd.read_csv(meta_csv)["Donor ID"].tolist()

"""Figures for the CPS association analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SEA-AD subclass order: excitatory, inhibitory, non-neuronal
SUBCLASS_ORDER = [
    "L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b",
    "Lamp5", "Lamp5 Lhx6", "Pax6", "Sncg", "Vip", "Sst", "Sst Chodl", "Pvalb", "Chandelier",
    "Astrocyte", "Microglia-PVM", "Oligodendrocyte", "OPC", "Endothelial", "VLMC & Perivascular",
]


def _order(groups) -> list[str]:
    known = [g for g in SUBCLASS_ORDER if g in set(groups)]
    return known + sorted(set(groups) - set(known))


def association_heatmap(res: pd.DataFrame, cols: list[str], path: Path, title: str = "", groupby: str = "subclass"):
    d = res[res["score"].isin(cols)]
    if d.empty:
        return None
    r = d.pivot(index="score", columns=groupby, values="r").reindex(index=cols)
    q = d.pivot(index="score", columns=groupby, values="q").reindex(index=cols)
    p = d.pivot(index="score", columns=groupby, values="p_perm").reindex(index=cols)
    order = _order(r.columns)
    r, q, p = r[order], q[order], p[order]
    fig, ax = plt.subplots(figsize=(0.42 * len(order) + 2.5, 0.38 * len(cols) + 1.8))
    im = ax.imshow(r.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if q.iat[i, j] < 0.1:
                ax.text(j, i, "**", ha="center", va="center", fontsize=8)
            elif p.iat[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), [c.split("_", 1)[1] for c in cols], fontsize=8)
    ax.set_title(title + "\n* p<0.05 (permutation)   ** q<0.1", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6, label="r")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def top_scatter(pb: pd.DataFrame, res: pd.DataFrame, path: Path, n: int = 6, groupby: str = "subclass"):
    top = res.head(n)
    if top.empty:
        return None
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6), squeeze=False)
    for ax, (_, row) in zip(axes[0], top.iterrows()):
        d = pb[pb[groupby] == row[groupby]]
        ax.scatter(d["cps"], d[row["score"]], c=d["cps"], cmap="viridis", s=28, edgecolor="k", lw=0.4)
        b = np.polyfit(d["cps"], d[row["score"]], 1)
        xs = np.linspace(d["cps"].min(), d["cps"].max(), 10)
        ax.plot(xs, np.polyval(b, xs), color="0.3", lw=1)
        ax.set_title(f"{row[groupby]}\n{row['score']}\nr={row['r']:.2f} p={row['p_perm']:.3g}", fontsize=8)
        ax.set_xlabel("CPS", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def concordance(conc: pd.DataFrame, path: Path, groupby: str = "subclass"):
    if conc.empty:
        return None
    m = conc.pivot(index="program", columns=groupby, values="r")
    m = m[_order(m.columns)]
    fig, ax = plt.subplots(figsize=(0.42 * m.shape[1] + 2.5, 0.38 * m.shape[0] + 1.5))
    im = ax.imshow(m.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=8)
    ax.set_title("RNA score vs promoter accessibility across donors (r)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def primary_forest(primary: pd.DataFrame, path: Path):
    """Partial r with q-values for the pre-specified hypotheses, RNA and ATAC side by side."""
    d = primary.copy()
    d["label"] = d["cells"].str.replace("Neuronal: ", "") + " | " + d["program"]
    labels = list(dict.fromkeys(d["label"]))
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(labels) + 1.2))
    mods = [m for m in ("rna", "atac") if (d["modality"] == m).any()]
    for mod in mods:
        color = {"rna": "#3b6fb6", "atac": "#d9822b"}[mod]
        off = 0.0 if len(mods) == 1 else {"rna": -0.15, "atac": 0.15}[mod]
        m = d[d["modality"] == mod].set_index("label").reindex(labels)
        y = np.arange(len(labels)) + off
        if "n_donors" in m:
            # Fisher-z 95% CI (approximate: ignores covariate degrees of freedom)
            se = 1 / np.sqrt(np.maximum(m["n_donors"].to_numpy(float) - 3, 1))
            z = np.arctanh(np.clip(m["r"].to_numpy(float), -0.999, 0.999))
            ax.hlines(y, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), color=color, lw=1.2, alpha=0.6)
        ax.scatter(m["r"], y, color=color, s=30, label=mod.upper(), zorder=3)
        for yi, (_, row) in zip(y, m.iterrows()):
            if pd.notna(row["q_primary"]) and row["q_primary"] < 0.1:
                ax.text(row["r"] + 0.04, yi, f"q={row['q_primary']:.2g}", va="center", fontsize=7)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("partial r with CPS (across donors), 95% CI")
    if len(mods) > 1:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Pre-specified hypotheses", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def pvalue_qq(res: pd.DataFrame, path: Path):
    """Observed vs expected -log10 p; a curve on the diagonal means no signal beyond chance."""
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for mod, color in (("rna_", "#3b6fb6"), ("atac_", "#d9822b")):
        p = np.sort(res.loc[res["score"].str.startswith(mod), "p_perm"].to_numpy())
        if len(p):
            e = (np.arange(1, len(p) + 1) - 0.5) / len(p)
            ax.scatter(-np.log10(e), -np.log10(p), s=8, color=color, label=f"{mod[:-1].upper()} (n={len(p)})")
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("expected -log10 p")
    ax.set_ylabel("observed -log10 p")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("All cell group x program tests", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def primary_scatter(tables: dict, primary: list[tuple[str, str, str]], path: Path, modality: str = "rna"):
    """Donor pseudobulk vs CPS for each pre-specified hypothesis."""
    n = len(primary)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.4 * nrow), squeeze=False)
    for ax, (grouping, grp, prog) in zip(axes.ravel(), primary):
        pb = tables[grouping][0]
        d = pb[pb[grouping] == grp]
        col = f"{modality}_{prog}"
        ax.scatter(d["cps"], d[col], c=d["cps"], cmap="viridis", s=24, edgecolor="k", lw=0.4)
        if len(d) > 2:
            b = np.polyfit(d["cps"], d[col], 1)
            xs = np.linspace(d["cps"].min(), d["cps"].max(), 10)
            ax.plot(xs, np.polyval(b, xs), color="0.3", lw=1)
        ax.set_title(f"{grp.replace('Neuronal: ', '')}\n{prog}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("CPS", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def region_heatmap(regions: pd.DataFrame, path: Path, order: list[str] | None = None):
    """Pre-specified hypotheses (rows) x brain regions (columns), coloured by partial r with local CPS."""
    d = regions.copy()
    d["label"] = d["cells"].str.replace("Neuronal: ", "") + " | " + d["program"]
    labels = list(dict.fromkeys(d["label"]))
    cols = [r for r in (order or sorted(d["region"].unique())) if r in set(d["region"])]
    r = d.pivot(index="label", columns="region", values="r").reindex(index=labels, columns=cols)
    q = d.pivot(index="label", columns="region", values="q_primary").reindex(index=labels, columns=cols)
    p = d.pivot(index="label", columns="region", values="p_perm").reindex(index=labels, columns=cols)
    n = d.groupby("region")["n_donors_region"].first().reindex(cols)
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 3.2, 0.36 * len(labels) + 1.6))
    im = ax.imshow(r.to_numpy(float), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if q.iat[i, j] < 0.1:
                ax.text(j, i, "**", ha="center", va="center", fontsize=8)
            elif p.iat[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), [f"{c}\n(n={int(k)})" for c, k in zip(cols, n)], fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Partial r with local CPS by region   * p<0.05   ** q<0.1 (within region)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6, label="r")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def concordance_null(obs_r: np.ndarray, null_medians: np.ndarray, path: Path):
    """Distribution of RNA-ATAC concordance r vs. the donor-shuffled null of the median."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(obs_r, bins=24, color="#3b6fb6", alpha=0.8, label=f"observed (n={len(obs_r)})")
    ax.axvline(np.median(obs_r), color="#3b6fb6", lw=1.5)
    lo, hi = np.quantile(null_medians, [0.025, 0.975])
    ax.axvspan(lo, hi, color="0.6", alpha=0.35, label="null median, 95% range")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("r: RNA program score vs promoter accessibility (across donors)")
    ax.set_ylabel("cell group x program")
    ax.set_title(f"Multiome concordance: median r = {np.median(obs_r):.2f}", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig

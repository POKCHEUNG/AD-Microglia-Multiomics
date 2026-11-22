"""Gene sets. DDR pathways, p53 targets and senescence are reused from ``ddrmulti``;
AD / glial programs are added here.

DAM: Keren-Shaul et al. 2017 Cell. cGAS-STING / type-I IFN: core sensors + canonical ISGs.
Reactive astrocytes: Escartin et al. 2021 Nat Neurosci consensus markers.
"""

from ddrmulti.signatures import DDR_PATHWAYS, FATE_PROGRAMS, P53_TARGETS

CGAS_STING_IFN = ["CGAS", "STING1", "TBK1", "IRF3", "IRF7", "IFI16", "STAT1", "ISG15", "IFIT1",
                  "IFIT3", "IFI44L", "MX1", "OAS1", "IFITM3"]
DAM = ["TREM2", "APOE", "TYROBP", "SPP1", "LPL", "CST7", "CD9", "ITGAX", "CLEC7A", "AXL",
       "CSF1", "LGALS3", "CTSD", "CTSB", "GPNMB"]
REACTIVE_ASTRO = ["GFAP", "VIM", "SERPINA3", "CHI3L1", "C3", "OSMR", "CD44", "S100B", "AQP4", "STAT3"]

# Repair pathways that operate in post-mitotic neurons (no replication-coupled HR / FA / MMR).
NEURONAL_REPAIR = ["NHEJ", "BER", "NER", "Checkpoint"]


def all_programs() -> dict[str, list[str]]:
    return {
        **DDR_PATHWAYS,
        "p53_targets": P53_TARGETS,
        "Senescence": FATE_PROGRAMS["Senescence"],
        "cGAS_STING_IFN": CGAS_STING_IFN,
        "DAM": DAM,
        "Reactive_astro": REACTIVE_ASTRO,
    }

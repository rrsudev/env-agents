"""
Run composite-score pipelines for one or many measures.

Basic examples
--------------
# Run every measure using the default prediction set (ALLCOND)
python run_measures.py

# Run only two measures (envactions and nep) with default predictions
python run_measures.py envactions nep


Agent-specific predictions
--------------------------
Add the --agent flag to swap in a different prediction file
(choices: ALLCONDITIONS, AMERICANVOICES, DEMOGRAPHIC, ENVIRONMENTAL).

# Climate/Environmental agent – run everything
python run_measures.py --agent ENVIRONMENTAL

# Demographic agent – run just CNS and NEP, saving to a custom folder
python run_measures.py cns nep --agent DEMOGRAPHIC --outdir results/demog

Flags
-----
measures   one or more pipeline names, or omit / 'all' for every measure  
--agent    which agent’s predictions to use [default: ALLCONDITIONS]  
--outdir   where to write the output CSVs  [default: ./apr8analysispipeline/00testing]
"""


import os
import sys
import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# GLOBAL CONFIG 
# ---------------------------------------------------------------------
# !!! TO THE USER: Make sure these are accurate to your own file structure
DATA_W1         = "groundtruth_data/w1_groundtruth.csv"
DATA_W2         = "groundtruth_data/w2_groundtruth.csv"
DATA_PRED       = "simulated_data/simulated_allcond.csv",

# pathways for agent conditions
# !!! TO THE USER: Make sure these are accurate to your own file structure
PRED_FILES = {
    "ALLCONDITIONS":        "simulated_data/simulated_allcond.csv",
    "AMERICANVOICES":       "simulated_data/simulated_americanvoices.csv",
    "DEMOGRAPHIC":          "simulated_data/simulated_demographic.csv",
    "ENVIRONMENTAL":        "simulated_data/simulated_allcond.csv",
}


def standardize_email(df: pd.DataFrame) -> pd.DataFrame:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    return df

# ---------------------------------------------------------------------
# --------------  MEASURE PIPELINES  ----------------------------------
# • Every measure-level function will:
#       1. accept (outdir: str) arg
#       2. save exactly ONE csv inside that outdir
#       3. return a short status string
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# --------------  ENVIRONMENTAL ACTIONS    ----------------------------
# ---------------------------------------------------------------------

def run_envactions(outdir: str) -> str:
    """Environmental Actions 17-item Likert composite."""
    env_cols = [f"ENV_ACTIONS_{i}" for i in range(1, 18)]
    likert = {
        "Very unlikely": 1, "Unlikely": 2, "Somewhat unlikely": 3,
        "Neutral": 4,
        "Somewhat likely": 5, "Likely": 6, "Very likely": 7,
        "I already do this": 7
    }

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    def map_scale(df, tag):
        for c in env_cols:
            if c in df.columns:
                df[c] = df[c].map(likert).astype(float)
        df[f"ENV_ACTIONS_Composite_{tag}"] = df[env_cols].mean(axis=1) / 7.0
        return df[["Email", f"ENV_ACTIONS_Composite_{tag}"]]

    df_w1   = map_scale(df_w1,   "W1")
    df_w2   = map_scale(df_w2,   "W2")
    df_pred = map_scale(df_pred, "Pred")

    merged = (
        df_w1
        .merge(df_w2,   on="Email")
        .merge(df_pred, on="Email")
        .dropna()
    )

    out_path = os.path.join(outdir, "envactions_output.csv")
    merged.to_csv(out_path, index=False)
    return f"ENV-ACTIONS → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  ADAPTATION & MITIGATION  ----------------------------
# ---------------------------------------------------------------------
def run_adaptation_mitigation(outdir: str) -> str:
    """
    14-item adaptation/mitigation scale.
    Saves:  adaptation_mitigation_output.csv   (one row per e-mail)
    """

    adaptation_cols = ["Adaptation_1", "Adaptation_2", "Adaptation_3"]
    mitigation_cols = [
        "Mitigation_1", "Mitigation_2", "Mitigation_3", "Mitigation_4",
        "Mitigation_5", "Mitigation_6", "Mitigation_7", "Mitigation_8",
        "Mitigation_9", "Mitigation_10", "Mitigation_11"
    ]
    reverse_mitigation = ["Mitigation_2", "Mitigation_4", "Mitigation_7", "Mitigation_9"]

    likert_map_7pt = {
        "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
        "Unsure": 4, "Neither agree nor disagree": 4,
        "Somewhat agree": 5, "Agree": 6, "Strongly agree": 7
    }

    def reverse_code(series: pd.Series, max_val: int = 7) -> pd.Series:
        return max_val + 1 - series

    def load_and_preprocess(csv_path: str) -> pd.DataFrame:
        df = standardize_email(pd.read_csv(csv_path))
        for col in adaptation_cols + mitigation_cols:
            if col in df.columns:
                df[col] = df[col].map(likert_map_7pt).astype(float)
        for col in reverse_mitigation:
            if col in df.columns:
                df[col] = reverse_code(df[col], 7)
        return df

    def truth_composites(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        adap = df[adaptation_cols].mean(axis=1)
        miti = df[mitigation_cols].mean(axis=1)
        overall = pd.concat([adap, miti], axis=1).mean(axis=1)
        return pd.DataFrame({
            "Email": df["Email"],
            f"Adaptation_{tag}": adap,
            f"Mitigation_{tag}": miti,
            f"Truth_Overall_{tag}": overall
        })

    def pred_composites(df: pd.DataFrame) -> pd.DataFrame:
        adap = df[adaptation_cols].mean(axis=1)
        miti = df[mitigation_cols].mean(axis=1)
        overall = pd.concat([adap, miti], axis=1).mean(axis=1)
        return pd.DataFrame({
            "Email": df["Email"],
            "Adaptation_Pred": adap,
            "Mitigation_Pred": miti,
            "Overall_Pred": overall
        })

    df_w1   = load_and_preprocess(DATA_W1)
    df_w2   = load_and_preprocess(DATA_W2)
    df_pred = load_and_preprocess(DATA_PRED)

    w1  = truth_composites(df_w1,  "W1")
    w2  = truth_composites(df_w2,  "W2")
    pr  = pred_composites(df_pred)

    merged = (
        w1.merge(w2, on="Email")
          .merge(pr, on="Email")
    )

    # normalise to 0-1 scale
    for c in merged.columns:
        if c != "Email":
            merged[c] = merged[c] / 7.0

    out_path = os.path.join(outdir, "adaptation_mitigation_output.csv")
    merged.to_csv(out_path, index=False)

    return f"ADAPT+MITIG → rows:{len(merged)}"

# ---------------------------------------------------------------------
# --------------  NEW ECONOLOGICAL PARADIGM (NEP)----------------------
# ---------------------------------------------------------------------
def run_nep(outdir: str) -> str:
    """
    NEP Scale (15-item 5-pt Likert with reverse-coded items).
    Saves: nep_output.csv with Email, NEP_Composite_W1/W2/Pred
    """
    nep_cols = [f"NEPS_{i}" for i in range(1, 16)]
    reverse_nep = ["NEPS_2", "NEPS_6", "NEPS_12"]
    likert_5pt = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Unsure": 3,
        "Agree": 4,
        "Strongly agree": 5
    }

    def reverse_code(series, max_val=5):
        return max_val + 1 - series

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    for df in [df_w1, df_w2, df_pred]:
        for col in nep_cols:
            if col in df.columns:
                df[col] = df[col].map(likert_5pt).astype(float)
        for col in reverse_nep:
            if col in df.columns:
                df[col] = reverse_code(df[col])

    df_w1["NEP_Composite_W1"]     = df_w1[nep_cols].mean(axis=1) / 5.0
    df_w2["NEP_Composite_W2"]     = df_w2[nep_cols].mean(axis=1) / 5.0
    df_pred["NEP_Composite_Pred"] = df_pred[nep_cols].mean(axis=1) / 5.0

    merged = (
        df_w1[["Email", "NEP_Composite_W1"]]
        .merge(df_w2[["Email", "NEP_Composite_W2"]], on="Email")
        .merge(df_pred[["Email", "NEP_Composite_Pred"]], on="Email")
    )

    out_path = os.path.join(outdir, "nep_output.csv")
    merged.to_csv(out_path, index=False)

    return f"NEP → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  PSYCH. PROXIMITY OF CLIMATE CHANGE ------------------
# ---------------------------------------------------------------------
def run_proximity(outdir: str) -> str:
    """
    Psychological Proximity (8-item distance ratings, reversed and normalized).
    Saves: proximity_output.csv with 0–1 normalized scores.
    """
    proximity_cols = [
        "PSYCHPROX_1_1", "PSYCHPROX_2_1", "PSYCHPROX_3_1", "PSYCHPROX_4_1",
        "PSYCHPROX_5_1", "PSYCHPROX_6_1", "PSYCHPROX_7_1", "PSYCHPROX_8_1"
    ]

    def reverse_proximity(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return 101 - numeric

    def compute_prox_composite(df, label):
        for col in proximity_cols:
            if col in df.columns:
                df[col] = reverse_proximity(df[col])
        raw_col = f"Proximity_Composite_{label}"
        norm_col = f"Proximity_{label}"
        df[raw_col] = df[proximity_cols].mean(axis=1)
        df[norm_col] = df[raw_col] / 100.0
        return df[["Email", norm_col]]

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    df_w1   = compute_prox_composite(df_w1,   "W1")
    df_w2   = compute_prox_composite(df_w2,   "W2")
    df_pred = compute_prox_composite(df_pred, "Pred")

    merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
    out_path = os.path.join(outdir, "proximity_output.csv")
    merged.to_csv(out_path, index=False)
    return f"PROXIMITY → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  CONNECTEDNESS TO NATURE        ----------------------
# ---------------------------------------------------------------------
def run_cns(outdir: str) -> str:
    """
    Connectedness to Nature Scale (CNS): 14 items, 7-point Likert.
    Reverse-coded: items 4, 12, 14. Normalized to [0,1].
    """
    likert_map = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Somewhat disagree": 3,
        "Neutral": 4,
        "Neither agree nor disagree": 4,
        "Somewhat agree": 5,
        "Agree": 6,
        "Strongly agree": 7
    }
    cns_cols = [f"CONNECTNATURE_{i}" for i in range(1, 15)]
    reverse_cns = ["CONNECTNATURE_4", "CONNECTNATURE_12", "CONNECTNATURE_14"]

    def reverse_code(series):
        return 8 - pd.to_numeric(series, errors="coerce")

    def preprocess(df):
        for col in cns_cols:
            if col in df.columns:
                df[col] = df[col].map(likert_map).astype(float)
        for col in reverse_cns:
            if col in df.columns:
                df[col] = reverse_code(df[col])
        return df

    def compute(df, label):
        raw = df[cns_cols].mean(axis=1)
        norm = raw / 7.0
        return pd.DataFrame({"Email": df["Email"], f"CNS_Composite_{label}": norm})

    df_w1   = compute(preprocess(standardize_email(pd.read_csv(DATA_W1))), "W1")
    df_w2   = compute(preprocess(standardize_email(pd.read_csv(DATA_W2))), "W2")
    df_pred = compute(preprocess(standardize_email(pd.read_csv(DATA_PRED))), "Pred")

    merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
    out_path = os.path.join(outdir, "cns_output.csv")
    merged.to_csv(out_path, index=False)
    return f"CNS → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  MORAL FOUNDATIONS QUESTIONS    ----------------------
# ---------------------------------------------------------------------
def run_mfq(outdir: str) -> str:
    """
    Moral Foundations Questionnaire (MFQ): 32 items (5 subscales), 0–5 relevance scale.
    Normalized to [0,1]. Includes both subscale and overall mean.
    """
    mfq_cols = [f"MFQ_1_{i}" for i in range(1, 17)] + [f"MFQ_2_{i}" for i in range(1, 17)]
    foundations = {
        "Harm_Care":   [1, 7, 12, 17, 23, 28],
        "Fairness":    [2, 8, 13, 18, 24, 29],
        "Loyalty":     [3, 9, 14, 19, 25, 30],
        "Authority":   [4, 10, 15, 20, 26, 31],
        "Sanctity":    [5, 11, 16, 21, 27, 32],
    }

    response_map = {
        '(0) Not at all relevant (This consideration has nothing to do with my judgments of right and wrong.)': 0,
        '(1) Not very relevant': 1,
        '(2) Slightly relevant': 2,
        '(3) Somewhat relevant': 3,
        '(4) Very relevant': 4,
        '(5) Extremely relevant (This is one of the most important factors when I judge right and wrong.)': 5,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
    }

    def map_foundations(df, label):
        for name, qnums in foundations.items():
            subcols = [f"MFQ_1_{q}" if q <= 16 else f"MFQ_2_{q - 16}" for q in qnums]
            df[f"MFQ_{name}_{label}"] = df[subcols].mean(axis=1)
        base = [f"MFQ_{f}_{label}" for f in foundations]
        df[f"MFQ_overall_{label}"] = df[base].mean(axis=1)
        return df[["Email"] + base + [f"MFQ_overall_{label}"]]

    def prepare(df, label):
        df = standardize_email(df)
        df[mfq_cols] = df[mfq_cols].map(response_map.get)
        return map_foundations(df, label)

    df_w1   = prepare(pd.read_csv(DATA_W1),   "W1")
    df_w2   = prepare(pd.read_csv(DATA_W2),   "W2")
    df_pred = prepare(pd.read_csv(DATA_PRED), "Pred")

    merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

    for col in merged.columns:
        if col != "Email":
            merged[col] = merged[col] / 5.0

    out_path = os.path.join(outdir, "mfq_output.csv")
    merged.to_csv(out_path, index=False)
    return f"MFQ → rows:{len(merged)}"

# ---------------------------------------------------------------------
# --------------  GENERAL SELF-EFFICACY (GSE)    ----------------------
# ---------------------------------------------------------------------
def run_gses(outdir: str) -> str:
    """
    Compute GSE composite (0–1) for W1, W2, Pred.
    Saves:  gses_output.csv
    """
    gse_cols = [f"GSES_{i}" for i in range(1, 11)]
    likert = {
        "(1) Not at all true": 1,
        "(2) Hardly true":     2,
        "(3) Moderately true": 3,
        "(4) Exactly true":    4,
    }

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    for df in (df_w1, df_w2, df_pred):
        for c in gse_cols:
            if c in df.columns:
                df[c] = df[c].map(likert).astype(float)

    def comp(df, tag):
        df[f"GSE_Composite_{tag}"] = df[gse_cols].sum(axis=1)
        df[f"GSE_Composite_{tag}"] = (df[f"GSE_Composite_{tag}"] - 10) / 30  # [0–1]
        return df[["Email", f"GSE_Composite_{tag}"]]

    df_w1   = comp(df_w1,   "W1")
    df_w2   = comp(df_w2,   "W2")
    df_pred = comp(df_pred, "Pred")

    merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
    out_path = os.path.join(outdir, "gses_output.csv")
    merged.to_csv(out_path, index=False)

    return f"GSES  → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  INTERPERSONAL REACTIVITY INDEX (IRI)   --------------
# ---------------------------------------------------------------------
def run_iri(outdir: str) -> str:
    """
    Compute IRI PT, FS, EC, PD sub-scales (0–1) for W1, W2, Pred.
    Saves:  iri_output.csv
    """
    iri_map = {
        "IRI_1":  ("FS", False), "IRI_2":  ("EC", False), "IRI_3":  ("PT", True),
        "IRI_4":  ("EC", True),  "IRI_5":  ("FS", False), "IRI_6":  ("PD", False),
        "IRI_7":  ("FS", True),  "IRI_8":  ("PT", False), "IRI_9":  ("EC", False),
        "IRI_10": ("PD", False), "IRI_11": ("PT", False), "IRI_12": ("FS", True),
        "IRI_13": ("PD", True),  "IRI_14": ("EC", True),  "IRI_15": ("PT", True),
        "IRI_16": ("FS", False), "IRI_17": ("PD", False), "IRI_18": ("EC", True),
        "IRI_19": ("PD", True),  "IRI_20": ("EC", False), "IRI_21": ("PT", False),
        "IRI_22": ("EC", False), "IRI_23": ("FS", False), "IRI_24": ("PD", False),
        "IRI_25": ("PT", False), "IRI_26": ("FS", False), "IRI_27": ("PD", False),
        "IRI_28": ("PT", False),
    }
    direct  = {"A":0,"B":1,"C":2,"D":3,"E":4}
    reverse = {"A":4,"B":3,"C":2,"D":1,"E":0}
    scales  = ["PT","FS","EC","PD"]

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    def score(df):
        for item,(s,rev) in iri_map.items():
            if item in df.columns:
                df[item] = df[item].map(reverse if rev else direct).astype(float)
        return df

    df_w1   = score(df_w1)
    df_w2   = score(df_w2)
    df_pred = score(df_pred)

    def comp(df, tag):
        for s in scales:
            items = [i for i,(sub,_) in iri_map.items() if sub==s]
            df[f"IRI_{s}_{tag}"] = df[items].sum(axis=1) / 28  # 0–1
        return df[["Email"]+[f"IRI_{s}_{tag}" for s in scales]]

    df_w1   = comp(df_w1,   "W1")
    df_w2   = comp(df_w2,   "W2")
    df_pred = comp(df_pred, "Pred")

    merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
    out_path = os.path.join(outdir, "iri_output.csv")
    merged.to_csv(out_path, index=False)

    return f"IRI   → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  NEED FOR CLOSURE (NFC)     --------------------------
# ---------------------------------------------------------------------
def run_nfc(outdir: str) -> str:
    """
    Compute NFC composite (0–1) for W1, W2, Pred.
    Saves:  nfc_output.csv
    """
    nfc_cols = [f"NFC_{i}" for i in range(1,16)]
    likert = {
        "(1) Completely disagree": 1, "-2":2,"(2)":2,"-3":3,"(3)":3,
        "-4":4,"(4)":4,"-5":5,"(5)":5,"(6) Completely agree": 6
    }

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    def comp(df, tag):
        for c in nfc_cols:
            if c in df.columns:
                df[c] = df[c].map(likert).astype(float)
        df[f"NFC_Composite_{tag}"] = df[nfc_cols].mean(axis=1)
        df[f"NFC_Composite_{tag}"] = (df[f"NFC_Composite_{tag}"] - 1) / 5  # 0–1
        return df[["Email", f"NFC_Composite_{tag}"]]

    df_w1   = comp(df_w1,   "W1")
    df_w2   = comp(df_w2,   "W2")
    df_pred = comp(df_pred, "Pred")

    merged = (
        df_w1.merge(df_w2, on="Email")
             .merge(df_pred, on="Email")
             .loc[lambda d: d["Email"].notna()]
    )
    out_path = os.path.join(outdir, "nfc_output.csv")
    merged.to_csv(out_path, index=False)

    return f"NFC   → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  ENVIRONMENTAL EFFICACY   ----------------------------
# ---------------------------------------------------------------------
def run_envefficacy(outdir: str) -> str:
    indiv_cols = [f"EFFICACY_IND_{i}" for i in range(1, 5)]
    collect_cols = [f"EFFICACY_COLLECTIVE_{i}" for i in range(1, 5)]
    all_cols = indiv_cols + collect_cols

    likert_map = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Neither agree nor disagree': 3,
        'Somewhat agree': 4,
        'Strongly agree': 5,
    }

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    for df in [df_w1, df_w2, df_pred]:
        for col in all_cols:
            df[col] = df[col].map(likert_map).astype(float)

    def comp(df, tag):
        df[f"Eff_Indiv_{tag}"] = df[indiv_cols].sum(axis=1)
        df[f"Eff_Collective_{tag}"] = df[collect_cols].sum(axis=1)
        return df[["Email", f"Eff_Indiv_{tag}", f"Eff_Collective_{tag}"]]

    df_w1 = comp(df_w1, "W1")
    df_w2 = comp(df_w2, "W2")
    df_pred = comp(df_pred, "Pred")

    def normalize(s):
        return (s - 4) / 16

    for df in [df_w1, df_w2, df_pred]:
        for col in df.columns:
            if col.startswith("Eff_"):
                df[col] = normalize(df[col])

    merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
    merged.to_csv(os.path.join(outdir, "envefficacy_match.csv"), index=False)
    return f"ENVEFF → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  SOCIAL DOMINANCE ORIENTATION (SDO)    ---------------
# ---------------------------------------------------------------------
def run_sdo(outdir: str) -> str:
    sdo_cols = [f"SDO7_{i}" for i in range(1, 9)]
    sdo_rev = [f"SDO7_{i}" for i in [3,4,7,8]]
    likert = {
        "(1) Strongly Oppose": 1, "(2) Somewhat Oppose": 2, "(3) Slightly Oppose": 3,
        "(4) Neutral": 4, "(5) Slightly Favor": 5, "(6) Somewhat Favor": 6, "(7) Strongly Favor": 7
    }

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    for df in [df_w1, df_w2, df_pred]:
        for col in sdo_cols:
            df[col] = df[col].map(lambda x: likert.get(str(x).strip(), np.nan))
        for col in sdo_rev:
            df[col] = 8 - df[col]

    df_w1["SDO_Composite_W1"] = (df_w1[sdo_cols].mean(axis=1) - 1) / 6
    df_w2["SDO_Composite_W2"] = (df_w2[sdo_cols].mean(axis=1) - 1) / 6
    df_pred["SDO_Composite_Pred"] = (df_pred[sdo_cols].mean(axis=1) - 1) / 6

    merged = df_w1[["Email", "SDO_Composite_W1"]].merge(
        df_w2[["Email", "SDO_Composite_W2"]], on="Email").merge(
        df_pred[["Email", "SDO_Composite_Pred"]], on="Email").dropna()

    merged.to_csv(os.path.join(outdir, "sdo_output.csv"), index=False)
    return f"SDO    → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  GENERAL SYSTEM JUSTIFICATION SCALE (GSJS) -----------
# ---------------------------------------------------------------------
def run_gsjs(outdir: str) -> str:
    gsjs_cols = [f"GSJS_{i}" for i in range(1, 9)]
    reverse = ["GSJS_3", "GSJS_7"]
    likert = {
        "Strongly disagree": 1, "Disagree": 2, "Moderately disagree": 3,
        "Mildly disagree": 4, "Neither agree nor disagree": 5,
        "Mildly agree": 6, "Moderately agree": 7, "Agree": 8, "Strongly Agree": 9,
    }

    df_w1   = standardize_email(pd.read_csv(DATA_W1))
    df_w2   = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    for df in [df_w1, df_w2, df_pred]:
        for col in gsjs_cols:
            df[col] = df[col].map(lambda x: likert.get(str(x).strip(), np.nan))
        for col in reverse:
            df[col] = 10 - df[col]

    df_w1["GSJS_Composite_W1"] = (df_w1[gsjs_cols].mean(axis=1) - 1) / 8
    df_w2["GSJS_Composite_W2"] = (df_w2[gsjs_cols].mean(axis=1) - 1) / 8
    df_pred["GSJS_Composite_Pred"] = (df_pred[gsjs_cols].mean(axis=1) - 1) / 8

    merged = df_w1[["Email", "GSJS_Composite_W1"]].merge(
        df_w2[["Email", "GSJS_Composite_W2"]], on="Email").merge(
        df_pred[["Email", "GSJS_Composite_Pred"]], on="Email")

    merged.to_csv(os.path.join(outdir, "gsjs_output.csv"), index=False)
    return f"GSJS   → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  TRUST IN SCIENCE & CLIMATE SCIENTISTS ---------------
# ---------------------------------------------------------------------
def run_trust(outdir: str) -> str:
    """
    Composite of 21 trust items (SS_Q4_1 to SS_Q4_21). 5-point scale.
    Normalized to [0,1].
    """
    trust_cols = [f"SS_Q4_{i}" for i in range(1, 22)]
    trust_mapping = {
        "None at all": 1,
        "A little": 2,
        "A moderate amount": 3,
        "A lot": 4,
        "A great deal": 5
    }

    def compute(df, label):
        for col in trust_cols:
            if col in df.columns:
                df[col] = df[col].map(trust_mapping)
        raw = df[trust_cols].mean(axis=1)
        norm = raw / 5.0
        return pd.DataFrame({"Email": df["Email"], f"Trust_Composite_{label}": norm})

    df_w1 = compute(standardize_email(pd.read_csv(DATA_W1)), "W1")
    df_w2 = compute(standardize_email(pd.read_csv(DATA_W2)), "W2")
    df_pred = compute(standardize_email(pd.read_csv(DATA_PRED)), "Pred")

    merged = df_w1.merge(df_w2, on="Email", how="outer").merge(df_pred, on="Email", how="outer")
    merged = merged.dropna(subset=["Trust_Composite_W1", "Trust_Composite_W2"], how="all")

    out_path = os.path.join(outdir, "trust_output.csv")
    merged.to_csv(out_path, index=False)
    return f"TRUST → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  ENVIRONMENTAL EMOTIONS    ---------------------------
# ---------------------------------------------------------------------
def run_emotions(outdir: str) -> str:
    emotion_cols = [f"Emotions_{i}" for i in range(1, 15)]

    def encode(row, cols):
        return [1.0 if row[c] == "yes" else 0.0 if row[c] == "no" else np.nan for c in cols]

    df_w1 = standardize_email(pd.read_csv(DATA_W1))
    df_w2 = standardize_email(pd.read_csv(DATA_W2))
    df_pred = standardize_email(pd.read_csv(DATA_PRED))

    for df in [df_w1, df_w2, df_pred]:
        for col in emotion_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

    merged = df_w1[["Email"] + emotion_cols].merge(
        df_w2[["Email"] + emotion_cols], on="Email", suffixes=("_W1", "_W2")
    ).merge(
        df_pred[["Email"] + emotion_cols], on="Email"
    )

    records = []
    for _, row in merged.iterrows():
        email = row["Email"]
        v1 = np.array(encode(row, [f"{c}_W1" for c in emotion_cols]), dtype=np.float64)
        v2 = np.array(encode(row, [f"{c}_W2" for c in emotion_cols]), dtype=np.float64)
        vp = np.array(encode(row, emotion_cols), dtype=np.float64)

        def safe_mean(x):
            return np.nan if np.isnan(x).all() else np.nanmean(x)

        records.append({
            "Email": email,
            "Envemotions_Composite_W1": round(safe_mean(v1), 3),
            "Envemotions_Composite_W2": round(safe_mean(v2), 3),
            "Envemotions_Composite_Pred": round(safe_mean(vp), 3)
        })

    out_df = pd.DataFrame(records)
    out_path = os.path.join(outdir, "emotions_output.csv")
    out_df.to_csv(out_path, index=False)
    return f"EMOTIONS → rows:{len(out_df)}"





# ---------------------------------------------------------------------
# --------------  RISK AVERSION            ----------------------------
# ---------------------------------------------------------------------
def run_risk_aversion(outdir: str) -> str:
    """
    Binary-coded 10-item Risk Aversion scale. Option A = 1, Option B = 0.
    Normalized to [0,1].
    """
    ra_cols = [f"RA_{i}" for i in range(1, 11)]

    def map_averse(val):
        val = str(val).lower()
        return 1 if "option a" in val else 0 if "option b" in val else np.nan

    def compute(df, label):
        for col in ra_cols:
            df[col] = df[col].map(map_averse)
        raw = df[ra_cols].sum(axis=1)
        norm = raw / 10.0
        return pd.DataFrame({"Email": df["Email"], f"RA_Composite_{label}": norm})

    df_w1 = compute(standardize_email(pd.read_csv(DATA_W1)), "W1")
    df_w2 = compute(standardize_email(pd.read_csv(DATA_W2)), "W2")
    df_pred = compute(standardize_email(pd.read_csv(DATA_PRED)), "Pred")

    merged = df_w1.merge(df_w2, on="Email", how="outer").merge(df_pred, on="Email", how="outer")
    merged = merged.dropna(subset=["RA_Composite_W1", "RA_Composite_W2"], how="all")

    out_path = os.path.join(outdir, "risk_aversion_output.csv")
    merged.to_csv(out_path, index=False)
    return f"RISK-AVERSION → rows:{len(merged)}"


# ---------------------------------------------------------------------
# --------------  MORAL EXPANSIVENESS SCALE (MES)    ------------------
# ---------------------------------------------------------------------
def run_mes(outdir: str) -> str:
    """
    Moral Expansiveness Scale (MES): 30 items. 0–3 mapped scale. Normalized over 90 max.
    """
    mes_cols = [f"MES_Q5_{i}" for i in range(1, 31)]
    mes_map = {
        "Inner Circle of Moral Concern": 3,
        "Outer Circle of Moral Concern": 2,
        "Fringes of Moral Concern": 1,
        "Outside the Moral Boundary": 0
    }

    def compute(df, label):
        for col in mes_cols:
            if col in df.columns:
                df[col] = df[col].map(mes_map)
        raw = df[mes_cols].sum(axis=1)
        norm = raw / 90.0
        return pd.DataFrame({"Email": df["Email"], f"MES_Composite_{label}": norm})

    df_w1 = compute(standardize_email(pd.read_csv(DATA_W1)), "W1")
    df_w2 = compute(standardize_email(pd.read_csv(DATA_W2)), "W2")
    df_pred = compute(standardize_email(pd.read_csv(DATA_PRED)), "Pred")

    merged = df_w1.merge(df_w2, on="Email", how="outer").merge(df_pred, on="Email", how="outer")
    merged = merged.dropna(subset=["MES_Composite_W1", "MES_Composite_W2"], how="all")

    out_path = os.path.join(outdir, "mes_composite_output.csv")
    merged.to_csv(out_path, index=False)
    return f"MES → rows:{len(merged)}"


# -------------------------------------------------
# EVERY MEASURE PIPELINE AND EQUIVALENT FUNCTION
# map CLI-name  ->  function
# -------------------------------------------------
MEASURE_FUNCS = {
    "envactions":             run_envactions,
    "adaptation_mitigation":  run_adaptation_mitigation,
    "nep":                    run_nep,
    "proximity":              run_proximity,
    "cns":                    run_cns,
    "mfq":                    run_mfq,
    "gses":                   run_gses,
    "iri":                    run_iri,
    "nfc":                    run_nfc,
    "envefficacy":            run_envefficacy,
    "sdo":                    run_sdo,
    "gsjs":                   run_gsjs,
    "trust":                  run_trust,
    "envemotions":            run_emotions,
    "risk_aversion":          run_risk_aversion,
    "mes":                    run_mes,
}



# -----------------------------------------------------------------
# --------------  MAIN DISPATCHER  --------------------------------
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run composite-score pipelines.")

    parser.add_argument(
        "measures",
        nargs="*",
        help="'all' or one/more of: " + ", ".join(MEASURE_FUNCS.keys())
    )

    parser.add_argument(
        "--outdir",
        default=None, 
        help="Where to write output csvs (default: ./Composite_Outcomes/{AGENT}_waves)"
    )

    parser.add_argument(      
        "--agent",
        choices=list(PRED_FILES),  
        default="ALLCONDITIONS",
        help=("Which agent-type predictions to use "
              f"(default: ALLCONDITIONS). Choices: {', '.join(PRED_FILES)}")
    )

    args = parser.parse_args()

    global DATA_PRED
    DATA_PRED = PRED_FILES[args.agent]

    # Set output directory dynamically if not provided
    outdir = args.outdir or f"Composite_Outcomes/{args.agent}_waves"
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== AGENT CONDITION: {args.agent} ===")
    print(f"Using predictions from: {DATA_PRED}")
    print(f"Output directory: {outdir}\n")

    # Determine which measures to run
    targets = (
        list(MEASURE_FUNCS)
        if (not args.measures) or ("all" in args.measures)
        else args.measures
    )

    unknown = [m for m in targets if m not in MEASURE_FUNCS]
    if unknown:
        sys.exit(f"Unknown measure(s): {', '.join(unknown)}")

    for m in targets:
        print(f"→ Running {m} …")
        print("   ", MEASURE_FUNCS[m](outdir))

    print("\n✅ DONE – outputs saved in:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()

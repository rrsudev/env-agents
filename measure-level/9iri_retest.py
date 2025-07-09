import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/iri_output.csv"

# IRI mapping
iri_mapping = {
    "IRI_1":  ("FS", False), "IRI_2":  ("EC", False), "IRI_3":  ("PT", True),  "IRI_4":  ("EC", True),
    "IRI_5":  ("FS", False), "IRI_6":  ("PD", False), "IRI_7":  ("FS", True),  "IRI_8":  ("PT", False),
    "IRI_9":  ("EC", False), "IRI_10": ("PD", False), "IRI_11": ("PT", False), "IRI_12": ("FS", True),
    "IRI_13": ("PD", True),  "IRI_14": ("EC", True),  "IRI_15": ("PT", True),  "IRI_16": ("FS", False),
    "IRI_17": ("PD", False), "IRI_18": ("EC", True),  "IRI_19": ("PD", True),  "IRI_20": ("EC", False),
    "IRI_21": ("PT", False), "IRI_22": ("EC", False), "IRI_23": ("FS", False), "IRI_24": ("PD", False),
    "IRI_25": ("PT", False), "IRI_26": ("FS", False), "IRI_27": ("PD", False), "IRI_28": ("PT", False),
}

# Scoring maps
direct_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
reverse_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

# IRI subscales
scales = ["PT", "FS", "EC", "PD"]

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email keys
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Scoring function
def score_iri(df):
    for item, (subscale, reverse) in iri_mapping.items():
        if item in df.columns:
            df[item] = df[item].map(reverse_map if reverse else direct_map).astype(float)
    return df

df_w1 = score_iri(df_w1)
df_w2 = score_iri(df_w2)
df_pred = score_iri(df_pred)

# Composite computation
def compute_composites(df, label):
    for scale in scales:
        items = [item for item, (s, _) in iri_mapping.items() if s == scale]
        df[f"IRI_{scale}_{label}"] = df[items].sum(axis=1) / 28  # Normalize
    return df[["Email"] + [f"IRI_{s}_{label}" for s in scales]]

df_w1 = compute_composites(df_w1, "W1")
df_w2 = compute_composites(df_w2, "W2")
df_pred = compute_composites(df_pred, "Pred")

# Merge all
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Save
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

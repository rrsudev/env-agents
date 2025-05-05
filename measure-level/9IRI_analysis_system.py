import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/aggregated_outputs/iri_output.csv"

# IRI mapping based on chart
iri_mapping = {
    "IRI_1":  ("FS", False), "IRI_2":  ("EC", False), "IRI_3":  ("PT", True),  "IRI_4":  ("EC", True),
    "IRI_5":  ("FS", False), "IRI_6":  ("PD", False), "IRI_7":  ("FS", True),  "IRI_8":  ("PT", False),
    "IRI_9":  ("EC", False), "IRI_10": ("PD", False), "IRI_11": ("PT", False), "IRI_12": ("FS", True),
    "IRI_13": ("PD", True),  "IRI_14": ("EC", True),  "IRI_15": ("PT", True),  "IRI_16": ("FS", False),
    "IRI_17": ("PD", False), "IRI_18": ("EC", True),  "IRI_19": ("PD", True),  "IRI_20": ("EC", False),
    "IRI_21": ("PT", False), "IRI_22": ("EC", False), "IRI_23": ("FS", False), "IRI_24": ("PD", False),
    "IRI_25": ("PT", False), "IRI_26": ("FS", False), "IRI_27": ("PD", False), "IRI_28": ("PT", False),
}

# scoring map
direct_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
reverse_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

# load and preprocess
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# scoring function
def score_iri(df):
    for item, (subscale, reverse) in iri_mapping.items():
        if item in df.columns:
            if reverse:
                df[item] = df[item].map(reverse_map).astype(float)
            else:
                df[item] = df[item].map(direct_map).astype(float)
    return df

df_truth = score_iri(df_truth)
df_pred = score_iri(df_pred)

# calculate sub-composites based on category mappings
scales = ["PT", "FS", "EC", "PD"]

def compute_composites(df, label):
    for scale in scales:
        scale_items = [item for item, (s, _) in iri_mapping.items() if s == scale]
        df[f"IRI_{scale}_{label}"] = df[scale_items].sum(axis=1)
    return df

df_truth = compute_composites(df_truth, "truth")
df_pred = compute_composites(df_pred, "pred")

# normalize
for scale in scales:
    df_truth[f"IRI_{scale}_truth"] = df_truth[f"IRI_{scale}_truth"] / 28
    df_pred[f"IRI_{scale}_pred"] = df_pred[f"IRI_{scale}_pred"] / 28

merged = pd.merge(
    df_truth[["Email"] + [f"IRI_{scale}_truth" for scale in scales]],
    df_pred[["Email"] + [f"IRI_{scale}_pred" for scale in scales]],
    on="Email"
)
merged = merged[merged["Email"].notna()]

# save as csv
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved normalized IRI composite results to: {OUTPUT_PATH}")

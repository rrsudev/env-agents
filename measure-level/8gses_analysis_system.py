import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/aggregated_outputs/gse_individual_level_summary.csv"

# gse columns
gse_cols = [f"GSES_{i}" for i in range(1, 11)]

# mapping for 4-point scale
gse_likert_map = {
    "(1) Not at all true": 1,
    "(2) Hardly true": 2,
    "(3) Moderately true": 3,
    "(4) Exactly true": 4
}

# load and preprocess data 
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# map GSE columns to numeric vals
for col in gse_cols:
    df_truth[col] = df_truth[col].map(gse_likert_map).astype(float)
    df_pred[col] = df_pred[col].map(gse_likert_map).astype(float)

# calculate composites
df_truth["GSE_Composite"] = df_truth[gse_cols].sum(axis=1)
df_pred["GSE_Composite"] = df_pred[gse_cols].sum(axis=1)

# normalize
df_truth["GSE_Composite"] = (df_truth["GSE_Composite"] - 10) / (40 - 10)
df_pred["GSE_Composite"] = (df_pred["GSE_Composite"] - 10) / (40 - 10)

merged = pd.merge(
    df_truth[["Email", "GSE_Composite"]],
    df_pred[["Email", "GSE_Composite"]],
    on="Email", suffixes=("_Truth", "_Pred")
)

# save as csv
merged = merged[["Email", "GSE_Composite_Truth", "GSE_Composite_Pred"]]
merged.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Saved normalized GSE composite scores to: {OUTPUT_PATH}")

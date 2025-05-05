import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/DEMOGRAPHICaggregated_outputs/nfc_output.csv"

# column mapping
nfc_cols = [f"NFC_{i}" for i in range(1, 16)]

# likert mapping for 6 point scale
nfc_likert_map = {
    "(1) Completely disagree": 1,
    "-2": 2,
    "-3": 3,
    "-4": 4,
    "-5": 5,
    "(6) Completely agree": 6
}

# load data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# map NFC items to numerical values
for col in nfc_cols:
    df_truth[col] = df_truth[col].map(nfc_likert_map).astype(float)
    df_pred[col] = df_pred[col].map(nfc_likert_map).astype(float)

# calculate composites (mean of 15 items) ===
df_truth["NFC_Composite_Truth"] = df_truth[nfc_cols].mean(axis=1)
df_pred["NFC_Composite_Pred"] = df_pred[nfc_cols].mean(axis=1)

# normalize 
df_truth["NFC_Composite_Truth"] = (df_truth["NFC_Composite_Truth"] - 1) / 5
df_pred["NFC_Composite_Pred"] = (df_pred["NFC_Composite_Pred"] - 1) / 5

merged = pd.merge(
    df_truth[["Email", "NFC_Composite_Truth"]],
    df_pred[["Email", "NFC_Composite_Pred"]],
    on="Email"
)
merged = merged[merged["Email"].notna()]

# save as csv
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved normalized NFC composite results to: {OUTPUT_PATH}")

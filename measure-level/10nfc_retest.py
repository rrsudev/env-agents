import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/nfc_output.csv"

# NFC item columns
nfc_cols = [f"NFC_{i}" for i in range(1, 16)]

# Likert mapping for 6-point scale
nfc_likert_map = {
    "(1) Completely disagree": 1,
    "-2": 2,
    "(2)": 2,
    "-3": 3,
    "(3)": 3,
    "-4": 4,
    "(4)": 4,
    "-5": 5,
    "(5)": 5,
    "(6) Completely agree": 6
}

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize emails
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Apply mapping and normalize
def process_nfc(df, label):
    for col in nfc_cols:
        if col in df.columns:
            df[col] = df[col].map(nfc_likert_map).astype(float)
    df[f"NFC_Composite_{label}"] = df[nfc_cols].mean(axis=1)
    df[f"NFC_Composite_{label}"] = (df[f"NFC_Composite_{label}"] - 1) / 5  # normalize to [0, 1]
    return df[["Email", f"NFC_Composite_{label}"]]

df_w1 = process_nfc(df_w1, "W1")
df_w2 = process_nfc(df_w2, "W2")
df_pred = process_nfc(df_pred, "Pred")

# Merge all
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
merged = merged[merged["Email"].notna()]

# Save
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

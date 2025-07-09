import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/gses_output.csv"

# GSE columns
gse_cols = [f"GSES_{i}" for i in range(1, 11)]

# Likert mapping for 4-point scale
gse_likert_map = {
    "(1) Not at all true": 1,
    "(2) Hardly true": 2,
    "(3) Moderately true": 3,
    "(4) Exactly true": 4
}

# Load and preprocess
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize emails
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Apply mapping and convert GSE columns
for df in [df_w1, df_w2, df_pred]:
    for col in gse_cols:
        if col in df.columns:
            df[col] = df[col].map(gse_likert_map).astype(float)

# Compute composite scores and normalize to [0, 1]
def compute_gse(df, label):
    df[f"GSE_Composite_{label}"] = df[gse_cols].sum(axis=1)
    df[f"GSE_Composite_{label}"] = (df[f"GSE_Composite_{label}"] - 10) / 30  # range: [10, 40]
    return df[["Email", f"GSE_Composite_{label}"]]

df_w1 = compute_gse(df_w1, "W1")
df_w2 = compute_gse(df_w2, "W2")
df_pred = compute_gse(df_pred, "Pred")

# Merge all
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Save
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

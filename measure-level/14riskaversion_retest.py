import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
OUTPUT_PATH = "0env_fixed/risk_aversion_output.csv"

# Risk aversion columns
ra_cols = [f"RA_{i}" for i in range(1, 11)]
MAX_SCORE = 10  # Normalize over 10 questions

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Map responses to binary (1 = risk averse, 0 = risk seeking)
def map_to_risk_averse(val):
    if isinstance(val, str):
        val = val.lower()
        if "option a" in val:
            return 1
        elif "option b" in val:
            return 0
    return np.nan

for df in [df_w1, df_w2, df_pred]:
    for col in ra_cols:
        df[col] = df[col].map(map_to_risk_averse)

# Compute normalized composite score
def compute_composite(df, label):
    raw_col = f"RA_Composite_{label}"
    norm_col = f"RA_Composite_{label}"
    df[raw_col] = df[ra_cols].sum(axis=1)
    df[norm_col] = df[raw_col] / MAX_SCORE
    return df[["Email", norm_col]]

# Compute for all datasets
df_w1 = compute_composite(df_w1, "W1")
df_w2 = compute_composite(df_w2, "W2")
df_pred = compute_composite(df_pred, "Pred")

# Merge all on Email
merged = df_w1.merge(df_w2, on="Email", how="outer").merge(df_pred, on="Email", how="outer")

# Drop people without ground truth
merged = merged.dropna(subset=["RA_Composite_W1", "RA_Composite_W2"], how="all")

# Save
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Normalized Risk Aversion summary saved to: {OUTPUT_PATH}")

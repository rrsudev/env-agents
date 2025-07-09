import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
OUTPUT_CSV = "0env_fixed/proximity_output.csv"

# Proximity items
proximity_cols = [
    "PSYCHPROX_1_1", "PSYCHPROX_2_1", "PSYCHPROX_3_1", "PSYCHPROX_4_1",
    "PSYCHPROX_5_1", "PSYCHPROX_6_1", "PSYCHPROX_7_1", "PSYCHPROX_8_1"
]

# Reverse code function
def reverse_proximity(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return 101 - numeric

# Normalize composite between 0–1
def compute_proximity_composite(df, label):
    for col in proximity_cols:
        if col in df.columns:
            df[col] = reverse_proximity(df[col])
    raw_col = f"Proximity_Composite_{label}"
    norm_col = f"Proximity_{label}"
    df[raw_col] = df[proximity_cols].mean(axis=1)
    df[norm_col] = df[raw_col] / 100  # Normalize 0–1
    return df[["Email", norm_col]]

# Load and standardize
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Compute normalized composites
df_w1 = compute_proximity_composite(df_w1, "W1")
df_w2 = compute_proximity_composite(df_w2, "W2")
df_pred = compute_proximity_composite(df_pred, "Pred")

# Merge and save
merged_composite = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
merged_composite.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved normalized proximity composite to: {OUTPUT_CSV}")

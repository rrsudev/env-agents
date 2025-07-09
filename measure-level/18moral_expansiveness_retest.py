import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
OUTPUT_CSV = "0env_fixed/mes_composite_output.csv"

# MES question columns
mes_cols = [f"MES_Q5_{i}" for i in range(1, 31)]

# Mapping from response category to score
mes_mapping = {
    "Inner Circle of Moral Concern": 3,
    "Outer Circle of Moral Concern": 2,
    "Fringes of Moral Concern": 1,
    "Outside the Moral Boundary": 0
}

MAX_SCORE = 30 * 3  # Normalize out of 90

# Load, map, and compute MES composite
def load_and_compute(csv_path, label):
    df = pd.read_csv(csv_path)
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

    for col in mes_cols:
        if col in df.columns:
            df[col] = df[col].map(mes_mapping)

    raw_col = f"MES_Composite_{label}"
    norm_col = f"MES_Composite_{label}"
    df[raw_col] = df[mes_cols].sum(axis=1, skipna=True)
    df[norm_col] = df[raw_col] / MAX_SCORE

    print(f"{label}: {df[raw_col].notna().sum()} MES scores computed")
    return df[["Email", norm_col]]

# Load all datasets
df_w1 = load_and_compute(WAVE1_CSV, "W1")
df_w2 = load_and_compute(WAVE2_CSV, "W2")
df_pred = load_and_compute(PREDICTED_CSV, "Pred")

# Merge on Email
merged = df_w1.merge(df_w2, on="Email", how="outer").merge(df_pred, on="Email", how="outer")

# Drop rows with no ground truth
merged = merged.dropna(subset=["MES_Composite_W1", "MES_Composite_W2"], how="all")

# Save output
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved normalized MES composite output to: {OUTPUT_CSV}")

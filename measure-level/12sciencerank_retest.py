import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
OUTPUT_CSV = "0env_fixed/trust_composite_output.csv"

# Full list of trust items
trust_cols = [f"SS_Q4_{i}" for i in range(1, 22)]
MAX_SCORE = 5  # Normalize by 5-point scale

# Mapping from text to numeric values
trust_mapping = {
    "None at all": 1,
    "A little": 2,
    "A moderate amount": 3,
    "A lot": 4,
    "A great deal": 5
}

# Normalize and compute composite score
def load_and_compute(csv_path, label):
    df = pd.read_csv(csv_path)
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

    # Apply mapping
    for col in trust_cols:
        if col in df.columns:
            df[col] = df[col].map(trust_mapping)

    raw_col = f"Trust_Composite_{label}"
    norm_col = f"Trust_Composite_{label}"
    df[raw_col] = df[trust_cols].mean(axis=1, skipna=True)
    df[norm_col] = df[raw_col] / MAX_SCORE

    print(f"{label}: {df[raw_col].notna().sum()} valid composites computed")
    return df[["Email", norm_col]]

# Load and compute for each dataset
df_w1 = load_and_compute(WAVE1_CSV, "W1")
df_w2 = load_and_compute(WAVE2_CSV, "W2")
df_pred = load_and_compute(PREDICTED_CSV, "Pred")

# Merge on Email
merged = df_w1.merge(df_w2, on="Email", how="outer").merge(df_pred, on="Email", how="outer")

# Drop rows with no ground truth
merged = merged.dropna(subset=["Trust_Composite_W1", "Trust_Composite_W2"], how="all")

# Save output
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Normalized trust composite output saved to: {OUTPUT_CSV}")

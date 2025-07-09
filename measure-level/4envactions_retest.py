import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/allcond_additional_measures/envactions_output.csv"

# Column mapping
env_cols = [f"ENV_ACTIONS_{i}" for i in range(1, 18)]

# Likert mapping for 7-point scale (+ override for "already do this")
likert_map_7pt = {
    "Very unlikely": 1,
    "Unlikely": 2,
    "Somewhat unlikely": 3,
    "Neutral": 4,
    "Somewhat likely": 5,
    "Likely": 6,
    "Very likely": 7,
    "I already do this": 7
}

# Load datasets
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email keys
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Map, normalize, and compute composite
def process_env(df, label):
    for col in env_cols:
        if col in df.columns:
            df[col] = df[col].map(likert_map_7pt).astype(float)
    df[f"ENV_ACTIONS_Composite_{label}"] = df[env_cols].mean(axis=1) / 7.0  # Normalize to [0,1]
    return df[["Email", f"ENV_ACTIONS_Composite_{label}"]]

df_w1 = process_env(df_w1, "W1")
df_w2 = process_env(df_w2, "W2")
df_pred = process_env(df_pred, "Pred")

# Merge all
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")
merged = merged.dropna(subset=[
    "ENV_ACTIONS_Composite_W1",
    "ENV_ACTIONS_Composite_W2",
    "ENV_ACTIONS_Composite_Pred"
])

# Compute correlations
r_w1_pred, _ = pearsonr(merged["ENV_ACTIONS_Composite_W1"], merged["ENV_ACTIONS_Composite_Pred"])
r_w1_w2, _ = pearsonr(merged["ENV_ACTIONS_Composite_W1"], merged["ENV_ACTIONS_Composite_W2"])
normalized_r = r_w1_pred / r_w1_w2 if r_w1_w2 != 0 else np.nan

# Output merged CSV
merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved ENV_ACTIONS: {OUTPUT_PATH}")

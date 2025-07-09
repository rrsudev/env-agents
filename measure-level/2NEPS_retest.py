import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/nep_output.csv"

# Likert mapping for 5-point scale
likert_map_5pt = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Unsure": 3,
    "Agree": 4,
    "Strongly agree": 5
}

# Columns
nep_cols = [f"NEPS_{i}" for i in range(1, 16)]
reverse_nep = ["NEPS_2", "NEPS_6", "NEPS_12"]

# Reverse code function
def reverse_code(series, max_value):
    return max_value + 1 - series

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email keys
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Map and reverse-code NEP items
for df in [df_w1, df_w2, df_pred]:
    for col in nep_cols:
        if col in df.columns:
            df[col] = df[col].map(likert_map_5pt).astype(float)
    for col in reverse_nep:
        if col in df.columns:
            df[col] = reverse_code(df[col], 5)

# Compute composite
df_w1["NEP_Composite_W1"] = df_w1[nep_cols].mean(axis=1)
df_w2["NEP_Composite_W2"] = df_w2[nep_cols].mean(axis=1)
df_pred["NEP_Composite_Pred"] = df_pred[nep_cols].mean(axis=1)

# Normalize to [0, 1]
df_w1["NEP_Composite_W1"] /= 5
df_w2["NEP_Composite_W2"] /= 5
df_pred["NEP_Composite_Pred"] /= 5

# Merge all
merged = df_w1[["Email", "NEP_Composite_W1"]].merge(
    df_w2[["Email", "NEP_Composite_W2"]], on="Email"
).merge(
    df_pred[["Email", "NEP_Composite_Pred"]], on="Email"
)

# Save as csv
merged.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved retest to: {OUTPUT_CSV}")

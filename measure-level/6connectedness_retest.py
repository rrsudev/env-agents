import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
OUTPUT_CSV    = "0env_fixed/cns_output.csv"

# Likert mapping for 7-point scale
likert_map_7pt = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Somewhat disagree": 3,
    "Neutral": 4,
    "Neither agree nor disagree": 4,
    "Somewhat agree": 5,
    "Agree": 6,
    "Strongly agree": 7
}

# Reverse code function
def reverse_code(series):
    return 8 - pd.to_numeric(series, errors="coerce")

# Recode direction (for future use if needed)
def recode_direction(val):
    val = pd.to_numeric(val, errors="coerce")
    if val in [6, 7]:
        return "Agree"
    elif val in [1, 2]:
        return "Disagree"
    elif val in [3, 4, 5]:
        return "Neutral"
    return np.nan

# Columns
cns_cols = [f"CONNECTNATURE_{i}" for i in range(1, 15)]
reverse_cns = ["CONNECTNATURE_4", "CONNECTNATURE_12", "CONNECTNATURE_14"]

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Likert mapping and reverse coding
def preprocess(df):
    for col in cns_cols:
        if col in df.columns:
            df[col] = df[col].map(likert_map_7pt).astype(float)
    for col in reverse_cns:
        if col in df.columns:
            df[col] = reverse_code(df[col])
    return df

df_w1 = preprocess(df_w1)
df_w2 = preprocess(df_w2)
df_pred = preprocess(df_pred)

# Composite calculation and normalization
def compute_cns_composite(df, label):
    raw_col = f"CNS_Composite_{label}"
    norm_col = f"CNS_Composite_{label}_Norm"
    df[raw_col] = df[cns_cols].mean(axis=1)
    df[norm_col] = df[raw_col] / 7  # Normalize to 0–1 range
    return df[["Email", norm_col]].rename(columns={norm_col: f"CNS_Composite_{label}"})

df_w1 = compute_cns_composite(df_w1, "W1")
df_w2 = compute_cns_composite(df_w2, "W2")
df_pred = compute_cns_composite(df_pred, "Pred")

# Merge all
agg_df = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Save
agg_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

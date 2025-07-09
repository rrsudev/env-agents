import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/envefficacy_match.csv"

# Columns
indiv_cols = [f"EFFICACY_IND_{i}" for i in range(1, 5)]
collect_cols = [f"EFFICACY_COLLECTIVE_{i}" for i in range(1, 5)]
all_cols = indiv_cols + collect_cols

# Likert mapping
likert_map_5 = {
    'Strongly disagree': 1,
    'Somewhat disagree': 2,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 4,
    'Strongly agree': 5
}

# Load and standardize
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    for col in all_cols:
        df[col] = df[col].map(likert_map_5).astype(float)

# Composite calculation
def compute_composites(df, label):
    df[f"Eff_Indiv_{label}"] = df[indiv_cols].sum(axis=1)
    df[f"Eff_Collective_{label}"] = df[collect_cols].sum(axis=1)
    return df[["Email", f"Eff_Indiv_{label}", f"Eff_Collective_{label}"]]

df_w1 = compute_composites(df_w1, "W1")
df_w2 = compute_composites(df_w2, "W2")
df_pred = compute_composites(df_pred, "Pred")

# Normalize to [0, 1]
def normalize(series):
    return (series - 4) / (20 - 4)

for df in [df_w1, df_w2, df_pred]:
    for col in df.columns:
        if col.startswith("Eff_") and col != "Email":
            df[col] = normalize(df[col])

# Merge all 
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Save
final_df = merged[[
    "Email",
    "Eff_Indiv_W1", "Eff_Indiv_W2", "Eff_Indiv_Pred",
    "Eff_Collective_W1", "Eff_Collective_W2", "Eff_Collective_Pred"
]]
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/gsjs_output.csv"

# GSJS columns and reverse-coded items
gsjs_cols = [f"GSJS_{i}" for i in range(1, 9)]
gsjs_reverse = ["GSJS_3", "GSJS_7"]

# 9-point Likert scale mapping
gsjs_scale_map = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Moderately disagree": 3,
    "Mildly disagree": 4,
    "Neither agree nor disagree": 5,
    "Mildly agree": 6,
    "Moderately agree": 7,
    "Agree": 8,
    "Strongly Agree": 9,
}

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize emails
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    df[gsjs_cols] = df[gsjs_cols].map(lambda x: gsjs_scale_map.get(str(x).strip(), np.nan))

# Reverse code GSJS_3 and GSJS_7
for df in [df_w1, df_w2, df_pred]:
    for col in gsjs_reverse:
        df[col] = 10 - df[col]  # 1 <-> 9

# Compute and normalize composite scores
df_w1["GSJS_Composite_W1"] = (df_w1[gsjs_cols].mean(axis=1) - 1) / 8
df_w2["GSJS_Composite_W2"] = (df_w2[gsjs_cols].mean(axis=1) - 1) / 8
df_pred["GSJS_Composite_Pred"] = (df_pred[gsjs_cols].mean(axis=1) - 1) / 8

# Merge all
output_df = df_w1[["Email", "GSJS_Composite_W1"]].merge(
    df_w2[["Email", "GSJS_Composite_W2"]], on="Email"
).merge(
    df_pred[["Email", "GSJS_Composite_Pred"]], on="Email"
)

# Save
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

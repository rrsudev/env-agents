import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/mfq_output.csv"

# MFQ columns based on subcategory chart
mfq_cols = [f"MFQ_1_{i}" for i in range(1, 17)] + [f"MFQ_2_{i}" for i in range(1, 17)]
mfq_foundations = {
    "Harm_Care":   [1, 7, 12, 17, 23, 28],
    "Fairness":    [2, 8, 13, 18, 24, 29],
    "Loyalty":     [3, 9, 14, 19, 25, 30],
    "Authority":   [4, 10, 15, 20, 26, 31],
    "Sanctity":    [5, 11, 16, 21, 27, 32],
}

# Likert mappings
response_map = {
    '(0) Not at all relevant (This consideration has nothing to do with my judgments of right and wrong.)': 0,
    '(1) Not very relevant': 1,
    '(2) Slightly relevant': 2,
    '(3) Somewhat relevant': 3,
    '(4) Very relevant': 4,
    '(5) Extremely relevant (This is one of the most important factors when I judge right and wrong.)': 5,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
}

# Load and preprocess
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    df[mfq_cols] = df[mfq_cols].map(response_map.get)

# Compute foundation scores
def compute_foundation_scores(df, label):
    for name, qnums in mfq_foundations.items():
        subcols = []
        for q in qnums:
            subcols.append(f"MFQ_1_{q}" if q <= 16 else f"MFQ_2_{q - 16}")
        df[f"MFQ_{name}_{label}"] = df[subcols].mean(axis=1)
    foundation_vars = [f"MFQ_{f}_{label}" for f in mfq_foundations]
    df[f"MFQ_overall_{label}"] = df[foundation_vars].mean(axis=1)
    return df[["Email"] + foundation_vars + [f"MFQ_overall_{label}"]]

df_w1 = compute_foundation_scores(df_w1, "W1")
df_w2 = compute_foundation_scores(df_w2, "W2")
df_pred = compute_foundation_scores(df_pred, "Pred")

# Merge all
merged = df_w1.merge(df_w2, on="Email").merge(df_pred, on="Email")

# Normalize to [0, 1]
for col in merged.columns:
    if col != "Email":
        merged[col] = merged[col] / 5

# Output columns
foundation_vars = list(mfq_foundations.keys())
final_cols = (
    ["Email"] +
    [f"MFQ_{f}_W1" for f in foundation_vars] +
    [f"MFQ_{f}_W2" for f in foundation_vars] +
    [f"MFQ_{f}_Pred" for f in foundation_vars] +
    ["MFQ_overall_W1", "MFQ_overall_W2", "MFQ_overall_Pred"]
)

# Save
merged[final_cols].to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/sdo_output.csv"

# SDO columns
sdo_cols = [f"SDO7_{i}" for i in range(1, 9)]
sdo_reverse = [f"SDO7_{i}" for i in [3, 4, 7, 8]]

# Likert mapping (7-point scale)
sdo_scale_map = {
    "(1) Strongly Oppose": 1,
    "(2) Somewhat Oppose": 2,
    "(3) Slightly Oppose": 3,
    "(4) Neutral": 4,
    "(5) Slightly Favor": 5,
    "(6) Somewhat Favor": 6,
    "(7) Strongly Favor": 7,
}

# Load and clean
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    for col in sdo_cols:
        df[col] = df[col].map(lambda x: sdo_scale_map.get(str(x).strip(), np.nan))
    for col in sdo_reverse:
        df[col] = 8 - df[col]

# Compute composites
df_w1["SDO_Composite_W1"] = (df_w1[sdo_cols].mean(axis=1) - 1) / 6
df_w2["SDO_Composite_W2"] = (df_w2[sdo_cols].mean(axis=1) - 1) / 6
df_pred["SDO_Composite_Pred"] = (df_pred[sdo_cols].mean(axis=1) - 1) / 6

# Merge all
final_df = df_w1[["Email", "SDO_Composite_W1"]].merge(
    df_w2[["Email", "SDO_Composite_W2"]], on="Email"
).merge(
    df_pred[["Email", "SDO_Composite_Pred"]], on="Email"
).dropna()

# Save
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

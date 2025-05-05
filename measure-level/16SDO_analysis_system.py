import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/DEMOGRAPHICaggregated_outputs/sdo_output.csv"

# map columns
sdo_cols = [f"SDO7_{i}" for i in range(1, 9)]
sdo_reverse = [f"SDO7_{i}" for i in [3, 4, 7, 8]]

# likert mapping 7 point scale
sdo_scale_map = {
    "(1) Strongly Oppose": 1,
    "(2) Somewhat Oppose": 2,
    "(3) Slightly Oppose": 3,
    "(4) Neutral": 4,
    "(5) Slightly Favor": 5,
    "(6) Somewhat Favor": 6,
    "(7) Strongly Favor": 7,
}

# load and preprocess data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

for col in sdo_cols:
    df_truth[col] = df_truth[col].map(lambda x: sdo_scale_map.get(str(x).strip(), np.nan))
    df_pred[col] = df_pred[col].map(lambda x: sdo_scale_map.get(str(x).strip(), np.nan))


# reverse code
for df in [df_truth, df_pred]:
    for col in sdo_reverse:
        df[col] = 8 - df[col]

# calculate composite and normalize
df_truth["SDO_Composite_Truth"] = (df_truth[sdo_cols].mean(axis=1) - 1) / 6
df_pred["SDO_Composite_Pred"] = (df_pred[sdo_cols].mean(axis=1) - 1) / 6

final_df = pd.merge(
    df_truth[["Email", "SDO_Composite_Truth"]],
    df_pred[["Email", "SDO_Composite_Pred"]],
    on="Email"
).dropna()

# save as csv
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved normalized SDO composite data to: {OUTPUT_PATH}")

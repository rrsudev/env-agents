import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_INDIVIDUALS = "apr8analysispipeline/aggregated_outputs/gsjs_output.csv"

# map columns
gsjs_cols = [f"GSJS_{i}" for i in range(1, 9)]
gsjs_reverse = ["GSJS_3", "GSJS_7"]

# scale mapping
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

# load and preprocess data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()
df_truth[gsjs_cols] = df_truth[gsjs_cols].applymap(lambda x: gsjs_scale_map.get(str(x).strip(), np.nan))
df_pred[gsjs_cols] = df_pred[gsjs_cols].applymap(lambda x: gsjs_scale_map.get(str(x).strip(), np.nan))

# reverse coding for items 3 and 7
for df in [df_truth, df_pred]:
    for col in gsjs_reverse:
        df[col] = 10 - df[col]  

# calculate composite score and normalize
df_truth["GSJS_Composite_Truth"] = (df_truth[gsjs_cols].mean(axis=1) - 1) / 8
df_pred["GSJS_Composite_Pred"] = (df_pred[gsjs_cols].mean(axis=1) - 1) / 8

output_df = pd.merge(
    df_truth[["Email", "GSJS_Composite_Truth"]],
    df_pred[["Email", "GSJS_Composite_Pred"]],
    on="Email",
    how="inner"
)

# save as csv
output_df.to_csv(OUTPUT_INDIVIDUALS, index=False)
print(f"\n✅ Saved GSJS composite scores (normalized) to: {OUTPUT_INDIVIDUALS}")

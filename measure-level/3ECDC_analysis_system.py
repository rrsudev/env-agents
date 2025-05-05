import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
AGG_OUTPUT = "apr8analysispipeline/DEMOGRAPHICaggregated_outputs/fixed_ecdc_output.csv"

# likert mapping for 5 points
likert_map_5pt = {
    "Strongly disagree": 1,
    "Somewhat agree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat disagree": 4,
    "Strongly agree": 5
}

# function for directional agreement
def recode_direction(val):
    if val in [4, 5]:
        return "Agree"
    elif val in [1, 2]:
        return "Disagree"
    elif val == 3:
        return "Neutral"
    return np.nan

# column mapping
indiv_cols = [f"ECDC_Individual_{i}" for i in range(1, 9)]
collect_cols = [f"ECDC_Collective_{i}" for i in range(1, 9)]

# load data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# map likert
for col in indiv_cols + collect_cols:
    df_truth[col] = df_truth[col].map(likert_map_5pt).astype(float)
    df_pred[col] = df_pred[col].map(likert_map_5pt).astype(float)

# calculate composite
df_truth["ECDC_Indiv_Truth"] = df_truth[indiv_cols].mean(axis=1)
df_truth["ECDC_Collect_Truth"] = df_truth[collect_cols].mean(axis=1)
df_pred["ECDC_Indiv_Pred"] = df_pred[indiv_cols].mean(axis=1)
df_pred["ECDC_Collect_Pred"] = df_pred[collect_cols].mean(axis=1)

# normalize
df_truth["ECDC_Indiv_Truth"] /= 5
df_truth["ECDC_Collect_Truth"] /= 5
df_pred["ECDC_Indiv_Pred"] /= 5
df_pred["ECDC_Collect_Pred"] /= 5

# save as csv
merged = pd.merge(
    df_truth[["Email", "ECDC_Indiv_Truth", "ECDC_Collect_Truth"]],
    df_pred[["Email", "ECDC_Indiv_Pred", "ECDC_Collect_Pred"]],
    on="Email", how="inner"
)
merged.to_csv(AGG_OUTPUT, index=False)
print(f"\n✅ Normalized composite scores saved to: {AGG_OUTPUT}")


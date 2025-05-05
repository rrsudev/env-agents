import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_CSV = "apr8analysispipeline/aggregated_outputs/adaptation_mitigation_output.csv"

# likert mapping for 7-point scales
likert_map_7pt = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Somewhat disagree": 3,
    "Unsure": 4,
    "Somewhat agree": 5,
    "Agree": 6,
    "Strongly agree": 7
}

# reverse coding
def reverse_code(series, max_value):
    return max_value + 1 - series

# categorize columns into adaptations, mitigations, reverse mitigations
adaptation_cols = ["Adaptation_1", "Adaptation_2", "Adaptation_3"]
mitigation_cols = [
    "Mitigation_1", "Mitigation_2", "Mitigation_3", "Mitigation_4", "Mitigation_5",
    "Mitigation_6", "Mitigation_7", "Mitigation_8", "Mitigation_9", "Mitigation_10", "Mitigation_11"
]
reverse_mitigation = ["Mitigation_2", "Mitigation_4", "Mitigation_7", "Mitigation_9"]

# load and preprocess data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

for col in adaptation_cols + mitigation_cols:
    df_truth[col] = df_truth[col].map(likert_map_7pt).astype(float)
    if col in df_pred.columns:
        df_pred[col] = df_pred[col].map(likert_map_7pt).astype(float)

for col in reverse_mitigation:
    df_truth[col] = reverse_code(df_truth[col], 7)
    if col in df_pred.columns:
        df_pred[col] = reverse_code(df_pred[col], 7)

# composites
df_truth["Adaptation_Composite_Truth"] = df_truth[adaptation_cols].mean(axis=1)
df_truth["Mitigation_Composite_Truth"] = df_truth[mitigation_cols].mean(axis=1)
df_truth["Overall_Composite_Truth"] = df_truth[["Adaptation_Composite_Truth", "Mitigation_Composite_Truth"]].mean(axis=1)

df_pred["Adaptation_Composite_Pred"] = df_pred[adaptation_cols].mean(axis=1)
df_pred["Mitigation_Composite_Pred"] = df_pred[mitigation_cols].mean(axis=1)
df_pred["Overall_Composite_Pred"] = df_pred[["Adaptation_Composite_Pred", "Mitigation_Composite_Pred"]].mean(axis=1)

# merge on email
merged = pd.merge(df_truth, df_pred, on="Email", how="inner")

# normalize
def normalize(series, scale_max=7):
    return series / scale_max

for col in [
    "Adaptation_Composite_Truth", "Adaptation_Composite_Pred",
    "Mitigation_Composite_Truth", "Mitigation_Composite_Pred",
    "Overall_Composite_Truth", "Overall_Composite_Pred"
]:
    merged[col] = normalize(merged[col])

# save as csv
merged[[
    "Email",
    "Adaptation_Composite_Truth", "Adaptation_Composite_Pred",
    "Mitigation_Composite_Truth", "Mitigation_Composite_Pred",
    "Overall_Composite_Truth", "Overall_Composite_Pred"
]].to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved normalized composite outputs to: {OUTPUT_CSV}")

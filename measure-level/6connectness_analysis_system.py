import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"

# likert mapping for 7-point scale
likert_map_7pt = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Somewhat disagree": 3,
    "Neutral": 4,
    "Somewhat agree": 5,
    "Agree": 6,
    "Strongly agree": 7
}

def reverse_code(series):
    return 8 - pd.to_numeric(series, errors="coerce")  # 1↔7, 2↔6, ..., 7↔1

def recode_direction(val):
    val = pd.to_numeric(val, errors="coerce")
    if val in [6, 7]:
        return "Agree"
    elif val in [1, 2]:
        return "Disagree"
    elif val in [3, 4, 5]:
        return "Neutral"
    return np.nan

# column mapping
cns_cols = [f"CONNECTNATURE_{i}" for i in range(1, 15)]
reverse_cns = ["CONNECTNATURE_4", "CONNECTNATURE_12", "CONNECTNATURE_14"]

# load and preprocessing data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# map likert
for col in cns_cols:
    if col in df_truth.columns:
        df_truth[col] = df_truth[col].map(likert_map_7pt).astype(float)
    if col in df_pred.columns:
        df_pred[col] = df_pred[col].map(likert_map_7pt).astype(float)

# reverse code
for col in reverse_cns:
    if col in df_truth.columns:
        df_truth[col] = reverse_code(df_truth[col])
    if col in df_pred.columns:
        df_pred[col] = reverse_code(df_pred[col])

# calculate composites
df_truth["CNS_Composite_Truth"] = df_truth[cns_cols].mean(axis=1)
df_pred["CNS_Composite_Pred"] = df_pred[cns_cols].mean(axis=1)

merged = pd.merge(
    df_truth[["Email", "CNS_Composite_Truth"]],
    df_pred[["Email", "CNS_Composite_Pred"]],
    on="Email",
    how="inner"
)
valid = merged.dropna()

df_truth["CNS_Composite_Truth_Norm"] = df_truth["CNS_Composite_Truth"] / 7
df_pred["CNS_Composite_Pred_Norm"] = df_pred["CNS_Composite_Pred"] / 7

# save as csv
agg_output = "apr8analysispipeline/aggregated_outputs/cns_output.csv"
agg_df = pd.merge(
    df_truth[["Email", "CNS_Composite_Truth_Norm"]],
    df_pred[["Email", "CNS_Composite_Pred_Norm"]],
    on="Email"
)
agg_df.to_csv(agg_output, index=False)
print(f"\n✅ Normalized CNS composite vectors saved to: {agg_output}")

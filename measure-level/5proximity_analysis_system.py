import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
AGG_OUTPUT = "apr8analysispipeline/aggregated_outputs/proximity_output.csv"

# columns
proximity_cols = [
    "PSYCHPROX_1_1", "PSYCHPROX_2_1", "PSYCHPROX_3_1", "PSYCHPROX_4_1",
    "PSYCHPROX_5_1", "PSYCHPROX_6_1", "PSYCHPROX_7_1", "PSYCHPROX_8_1"
]

# reverse coding function 
def reverse_proximity(series):
    numeric = pd.to_numeric(series, errors="coerce") 
    return 101 - numeric 

# load data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# reverse code and calculate composite
for col in proximity_cols:
    df_truth[col] = reverse_proximity(df_truth[col])
    df_pred[col] = reverse_proximity(df_pred[col])

df_truth["Proximity_Composite_Truth"] = df_truth[proximity_cols].mean(axis=1) / 100
df_pred["Proximity_Composite_Pred"] = df_pred[proximity_cols].mean(axis=1) / 100

# save as csv
merged_composite = pd.merge(
    df_truth[["Email", "Proximity_Composite_Truth"]],
    df_pred[["Email", "Proximity_Composite_Pred"]],
    on="Email", how="inner"
)
merged_composite.to_csv(AGG_OUTPUT, index=False)
print(f"\n✅ Normalized proximity composites saved to: {AGG_OUTPUT}")

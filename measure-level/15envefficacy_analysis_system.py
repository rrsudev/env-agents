import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/aggregated_outputs/efficacy_composite_match.csv"

# map columns
indiv_cols = [f"EFFICACY_IND_{i}" for i in range(1, 5)]
collect_cols = [f"EFFICACY_COLLECTIVE_{i}" for i in range(1, 5)]
all_cols = indiv_cols + collect_cols

# likert mapping for 5 point scale
likert_map_5 = {
    'Strongly disagree': 1,
    'Somewhat disagree': 2,
    'Neither agree nor disagree': 3,
    'Somewhat agree': 4,
    'Strongly agree': 5
}

# load and preprocess
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

for col in all_cols:
    df_truth[col] = df_truth[col].map(likert_map_5).astype(float)
    df_pred[col] = df_pred[col].map(likert_map_5).astype(float)

# compute composites
df_truth["Eff_Indiv"] = df_truth[indiv_cols].sum(axis=1)
df_truth["Eff_Collective"] = df_truth[collect_cols].sum(axis=1)
df_pred["Eff_Indiv"] = df_pred[indiv_cols].sum(axis=1)
df_pred["Eff_Collective"] = df_pred[collect_cols].sum(axis=1)

# normalize
def normalize(series):
    return (series - 4) / (20 - 4)

df_truth["Eff_Indiv"] = normalize(df_truth["Eff_Indiv"])
df_truth["Eff_Collective"] = normalize(df_truth["Eff_Collective"])
df_pred["Eff_Indiv"] = normalize(df_pred["Eff_Indiv"])
df_pred["Eff_Collective"] = normalize(df_pred["Eff_Collective"])

merged = pd.merge(
    df_truth[["Email", "Eff_Indiv", "Eff_Collective"]],
    df_pred[["Email", "Eff_Indiv", "Eff_Collective"]],
    on="Email", suffixes=("_truth", "_pred")
)

# save as csv
final_df = merged[["Email", "Eff_Indiv_truth", "Eff_Indiv_pred", "Eff_Collective_truth", "Eff_Collective_pred"]]
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Saved simplified normalized efficacy composites to: {OUTPUT_PATH}")

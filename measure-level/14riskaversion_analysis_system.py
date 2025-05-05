import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/aggregated_outputs/risk_aversion_output.csv"

# columns
ra_cols = [f"RA_{i}" for i in range(1, 11)]

# load and preprocess
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# clean choices
def map_choice(val):
    if isinstance(val, str):
        if "option a" in val.lower():
            return "A"
        elif "option b" in val.lower():
            return "B"
    return np.nan

for col in ra_cols:
    df_truth[col] = df_truth[col].map(map_choice)
    df_pred[col] = df_pred[col].map(map_choice)

merged = pd.merge(
    df_truth[["Email"] + ra_cols],
    df_pred[["Email"] + ra_cols],
    on="Email",
    suffixes=("_truth", "_pred")
)

# participant level analysis
results = []
for _, row in merged.iterrows():
    switch_point_truth = None
    switch_point_pred = None

    for i in range(10):
        t_val = row[f"RA_{i+1}_truth"]
        p_val = row[f"RA_{i+1}_pred"]

        if switch_point_truth is None and t_val == "B":
            switch_point_truth = i + 1
        if switch_point_pred is None and p_val == "B":
            switch_point_pred = i + 1

    switch_point_truth = switch_point_truth or 11  # If no switch, assume after 10
    switch_point_pred = switch_point_pred or 11

    direction_match = switch_point_truth == switch_point_pred

    results.append({
        "Email": row["Email"],
        "Switch_Point_Truth": switch_point_truth,
        "Switch_Point_Pred": switch_point_pred,
        "Exact_Switch_Match": direction_match
    })

# save as csv
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Risk Aversion results saved to: {OUTPUT_PATH}")

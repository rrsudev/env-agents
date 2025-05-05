import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_INDIVIDUALS = "apr8analysispipeline/DEMOGRAPHICaggregated_outputs/fixed_svi_output.csv"

# columns for SVI
svi_cols = [
    "SVI_Power", "SVI_Achievement", "SVI_Hedonism", "SVI_Stimulation", "SVI_SelfDirection",
    "SVI_Universalism", "SVI_Benevolence", "SVI_Tradition", "SVI_Conformity", "SVI_Security"
]

# likert scale mappings
# this takes into account unexpected behaviours from LLMs
svi_scale_map = {
    "Opposed to my principles (0)": 0,
    "Not important (1)": 1,
    "Important (4)": 4,
    "Of supreme importance (8)": 8,
    "-1": 1,
    "-2": 2,
    "-3": 3,
    "-4": 4,
    "-5": 5,
    "-6": 6,
    "-7": 7,
    "-8": 8,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
}

# mapping for directional agreement
def recode_direction(val):
    try:
        val = float(val)
        if val >= 6:
            return "High"
        elif val <= 2:
            return "Low"
        else:
            return "Moderate"
    except:
        return np.nan

# load and preprocess
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# map SVI to numeric scale
for col in svi_cols:
    df_truth[col] = df_truth[col].map(svi_scale_map).astype(float)
    df_pred[col] = df_pred[col].map(svi_scale_map).astype(float)

merged = pd.merge(
    df_truth[["Email"] + svi_cols],
    df_pred[["Email"] + svi_cols],
    on="Email",
    suffixes=("_truth", "_pred"),
    how="inner"
)

# per-person analysis
individual_results = []

for _, row in merged.iterrows():
    truths = []
    preds = []
    for col in svi_cols:
        t_val = row.get(f"{col}_truth")
        p_val = row.get(f"{col}_pred")
        if pd.notna(t_val) and pd.notna(p_val):
            truths.append(t_val)
            preds.append(p_val)

    if len(truths) > 1:  # need at least 2 points for Pearson
        r, _ = pearsonr(truths, preds)
        rmse = np.sqrt(np.mean((np.array(truths) - np.array(preds)) ** 2))
    else:
        r = np.nan
        rmse = np.nan

    exact_matches = sum(round(t) == round(p) for t, p in zip(truths, preds))
    directional_matches = sum(recode_direction(t) == recode_direction(p) for t, p in zip(truths, preds))
    total = len(truths)

    individual_results.append({
        "Email": row["Email"],
        "Pearson_r": round(r, 3) if pd.notna(r) else np.nan,
        "RMSE": round(rmse, 3) if pd.notna(rmse) else np.nan,
        "Exact_Match_Rate": round(exact_matches / total, 3) if total > 0 else np.nan,
        "Directional_Match_Rate": round(directional_matches / total, 3) if total > 0 else np.nan
    })

# save as csv
pd.DataFrame(individual_results).to_csv(OUTPUT_INDIVIDUALS, index=False)
print(f"\n✅ Saved individual-level SVI results to: {OUTPUT_INDIVIDUALS}")
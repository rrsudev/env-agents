import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
# OUTPUT_INDIVIDUALS = "apr8analysispipeline/aggregated_outputs/sciencerank_output.csv"
OUTPUT_NORMALIZED = "FIXED_SCIENCERANK/americanvoices_sciencerank_normalized_output.csv"


# mapping and subscales
ss_cols = {
    "Power": [f"SS_Q1_{i}" for i in range(1, 22)],
    "Influence": [f"SS_Q2_{i}" for i in range(1, 22)],
    "UseOfPower": [f"SS_Q3_{i}" for i in range(1, 22)],
    "Trust": [f"SS_Q4_{i}" for i in range(1, 22)],
    "ScienceLinks": [f"SS_Q5_{i}" for i in range(1, 11)],
    "Motives": [f"SS_Q6_{i}" for i in range(1, 12)]
}
all_items = sum(ss_cols.values(), [])

# science subscales
science_power_cols = ["SS_Q1_1", "SS_Q1_2", "SS_Q1_6", "SS_Q1_7", "SS_Q1_17"]
science_influence_cols = ["SS_Q2_1", "SS_Q2_2", "SS_Q2_6", "SS_Q2_7", "SS_Q2_17"]
science_usepower_cols = ["SS_Q3_1", "SS_Q3_2", "SS_Q3_6", "SS_Q3_7", "SS_Q3_17"]
science_trust_cols = ["SS_Q4_1", "SS_Q4_2", "SS_Q4_6", "SS_Q4_7", "SS_Q4_17"]

# load data
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

df_truth[all_items] = df_truth[all_items].apply(pd.to_numeric, errors="coerce")
df_pred[all_items] = df_pred[all_items].apply(pd.to_numeric, errors="coerce")

merged = pd.merge(
    df_truth[["Email"] + all_items],
    df_pred[["Email"] + all_items],
    on="Email",
    suffixes=("_truth", "_pred"),
    how="inner"
)

# per-person analysis
scale_lookup = {col: 10 if "Q3" in col else 5 for col in all_items}
individual_results = []

for _, row in merged.iterrows():
    exact_matches, total, dir_total, dir_matches = 0, 0, 0, 0
    truth_vals, pred_vals = [], []

    # science sub-sums
    power_truth = sum(row.get(f"{col}_truth", np.nan) for col in science_power_cols if pd.notna(row.get(f"{col}_truth", np.nan)))
    power_pred = sum(row.get(f"{col}_pred", np.nan) for col in science_power_cols if pd.notna(row.get(f"{col}_pred", np.nan)))

    influence_truth = sum(row.get(f"{col}_truth", np.nan) for col in science_influence_cols if pd.notna(row.get(f"{col}_truth", np.nan)))
    influence_pred = sum(row.get(f"{col}_pred", np.nan) for col in science_influence_cols if pd.notna(row.get(f"{col}_pred", np.nan)))

    usepower_truth = 0
    usepower_pred = 0
    for col in science_usepower_cols:
        t_val = row.get(f"{col}_truth", np.nan)
        p_val = row.get(f"{col}_pred", np.nan)
        if pd.notna(t_val):
            t_val_rescaled = (t_val - 1) / 9 * 4 + 1
            usepower_truth += t_val_rescaled
        if pd.notna(p_val):
            p_val_rescaled = (p_val - 1) / 9 * 4 + 1
            usepower_pred += p_val_rescaled

    trust_truth = sum(row.get(f"{col}_truth", np.nan) for col in science_trust_cols if pd.notna(row.get(f"{col}_truth", np.nan)))
    trust_pred = sum(row.get(f"{col}_pred", np.nan) for col in science_trust_cols if pd.notna(row.get(f"{col}_pred", np.nan)))

    # overall science rank sum
    overall_truth = power_truth + influence_truth + usepower_truth + trust_truth
    overall_pred = power_pred + influence_pred + usepower_pred + trust_pred

    # overall matching
    for col in all_items:
        t_val = row[f"{col}_truth"]
        p_val = row[f"{col}_pred"]
        if pd.notna(t_val) and pd.notna(p_val):
            truth_vals.append(t_val)
            pred_vals.append(p_val)
            total += 1
            if round(t_val) == round(p_val):
                exact_matches += 1

            def recode(val):
                scale = scale_lookup[col]
                if val >= scale * 0.75:
                    return "High"
                elif val <= scale * 0.25:
                    return "Low"
                else:
                    return "Moderate"

            t_dir = recode(t_val)
            p_dir = recode(p_val)
            if pd.notna(t_dir) and pd.notna(p_dir):
                dir_total += 1
                if t_dir == p_dir:
                    dir_matches += 1

    if total > 0:
        r = pearsonr(truth_vals, pred_vals)[0] if len(truth_vals) > 1 else np.nan
        rmse = np.sqrt(mean_squared_error(truth_vals, pred_vals))
        individual_results.append({
            "Email": row["Email"],
            "Science_Rank_Overall_Truth": round(overall_truth, 3),
            "Science_Rank_Overall_Pred": round(overall_pred, 3),
            "Pearson_r": round(r, 3) if pd.notna(r) else np.nan,
            "RMSE": round(rmse, 3),
            "Exact_Match_Rate": round(exact_matches / total, 3),
            "Directional_Match_Rate": round(dir_matches / dir_total, 3) if dir_total > 0 else np.nan
        })


df_out = pd.DataFrame(individual_results)

# normalize
scaler = MinMaxScaler()
cols_to_normalize = [col for col in df_out.columns if col not in ["Email", "Pearson_r", "RMSE", "Exact_Match_Rate", "Directional_Match_Rate"]]
df_out_normalized = df_out.copy()
df_out_normalized[cols_to_normalize] = scaler.fit_transform(df_out[cols_to_normalize])

# save as csv
df_out_normalized.to_csv(OUTPUT_NORMALIZED, index=False)
print(f"\n✅ Normalized sums saved to: {OUTPUT_NORMALIZED}")

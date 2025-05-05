import pandas as pd
import numpy as np

# CONFIG
GROUND_TRUTH_CSV = "apr8analysispipeline/apr8cleangroundtruth.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_americanvoices_responses.csv"
# PREDICTED_CSV = "apr8analysispipeline/processed_environment_responses.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_demographic_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/aggregated_outputs/envemotions_output.csv"

# column mappings
emotion_cols = [
    "Emotions_1", "Emotions_2", "Emotions_3", "Emotions_4", "Emotions_5",
    "Emotions_6", "Emotions_7", "Emotions_8", "Emotions_9", "Emotions_10",
    "Emotions_11", "Emotions_12", "Emotions_13", "Emotions_14"
]

# load data 
df_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)
df_truth["Email"] = df_truth["Email"].astype(str).str.lower().str.strip()
df_pred["Email"] = df_pred["Email"].astype(str).str.lower().str.strip()

# lowercase consistency
for col in emotion_cols:
    df_truth[col] = df_truth[col].astype(str).str.strip().str.lower()
    df_pred[col] = df_pred[col].astype(str).str.strip().str.lower()

merged = pd.merge(
    df_truth[["Email"] + emotion_cols],
    df_pred[["Email"] + emotion_cols],
    on="Email",
    suffixes=("_truth", "_pred"),
    how="inner"
)

# individual level analysis
individual_results = []

for _, row in merged.iterrows():
    total, exact_matches = 0, 0
    for col in emotion_cols:
        truth = row[f"{col}_truth"]
        pred = row[f"{col}_pred"]
        if truth in ["yes", "no"] and pred in ["yes", "no"]:
            total += 1
            if truth == pred:
                exact_matches += 1

    if total > 0:
        individual_results.append({
            "Email": row["Email"],
            "Emotion_Direct_Match_Rate": round(exact_matches / total, 3),
            "Items_Compared": total
        })


# save as csv
results_df = pd.DataFrame(individual_results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(results_df.sort_values(by="Emotion_Direct_Match_Rate").head(10).to_string(index=False))
print(f"\n✅ Saved to: {OUTPUT_PATH}")

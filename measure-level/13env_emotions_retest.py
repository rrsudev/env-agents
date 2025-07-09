import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_PATH = "apr8analysispipeline/allcond_additional_measures/envemotions_output.csv"

# Emotion item columns
emotion_cols = [
    "Emotions_1", "Emotions_2", "Emotions_3", "Emotions_4", "Emotions_5",
    "Emotions_6", "Emotions_7", "Emotions_8", "Emotions_9", "Emotions_10",
    "Emotions_11", "Emotions_12", "Emotions_13", "Emotions_14"
]

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email
for df in [df_w1, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()

# Normalize responses to lowercase
for df in [df_w1, df_pred]:
    for col in emotion_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

# Merge on Email
merged = df_w1[["Email"] + emotion_cols].merge(
    df_pred[["Email"] + emotion_cols],
    on="Email",
    suffixes=("_W1", "_Pred"),
    how="inner"
)

# Accuracy per item
item_accuracies = {}
for col in emotion_cols:
    matches = (merged[f"{col}_W1"] == merged[f"{col}_Pred"]) & merged[f"{col}_W1"].isin(["yes", "no"])
    total_valid = merged[[f"{col}_W1", f"{col}_Pred"]].apply(
        lambda x: x.iloc[0] in ["yes", "no"] and x.iloc[1] in ["yes", "no"],
        axis=1
    )
    if total_valid.sum() > 0:
        item_accuracies[col] = round(matches.sum() / total_valid.sum(), 3)
    else:
        item_accuracies[col] = np.nan

# Compute overall average accuracy
valid_accuracies = [v for v in item_accuracies.values() if not np.isnan(v)]
average_accuracy = round(sum(valid_accuracies) / len(valid_accuracies), 3) if valid_accuracies else np.nan

# Save to CSV
output_df = pd.DataFrame({
    "Item": list(item_accuracies.keys()),
    "Accuracy": list(item_accuracies.values())
})
output_df.loc[len(output_df.index)] = ["Average", average_accuracy]

output_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Climate emotion items: {OUTPUT_PATH}")

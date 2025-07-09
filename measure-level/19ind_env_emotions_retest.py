import pandas as pd
import numpy as np

# CONFIG
WAVE1_CSV = "jun1wave1_ground_truth.csv"
WAVE2_CSV = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_PATH = "00_acc_measure/allcond_ind_envemotions_output.csv"

# Emotion item columns
emotion_cols = [
    "Emotions_1", "Emotions_2", "Emotions_3", "Emotions_4", "Emotions_5",
    "Emotions_6", "Emotions_7", "Emotions_8", "Emotions_9", "Emotions_10",
    "Emotions_11", "Emotions_12", "Emotions_13", "Emotions_14"
]

# Load data
df_w1 = pd.read_csv(WAVE1_CSV)
df_w2 = pd.read_csv(WAVE2_CSV)
df_pred = pd.read_csv(PREDICTED_CSV)

# Standardize email and lowercase all emotion columns
for df in [df_w1, df_w2, df_pred]:
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    for col in emotion_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

# Merge all
merged = df_w1[["Email"] + emotion_cols].merge(
    df_w2[["Email"] + emotion_cols],
    on="Email",
    suffixes=("_W1", "_W2"),
    how="inner"
).merge(
    df_pred[["Email"] + emotion_cols],
    on="Email",
    how="inner"
)

# Compute individual-level accuracy for each pair
records = []

for _, row in merged.iterrows():
    participant_id = f"P{len(records) + 1:03d}"

    # Accuracy between W1 and W2 (human retest)
    w1_w2_matches = 0
    w1_w2_valid = 0

    # Accuracy between W1 and Prediction
    w1_pred_matches = 0
    w1_pred_valid = 0

    for col in emotion_cols:
        val_w1 = row[f"{col}_W1"]
        val_w2 = row[f"{col}_W2"]
        val_pred = row[col]  # prediction column kept as original name

        if val_w1 in ["yes", "no"] and val_w2 in ["yes", "no"]:
            w1_w2_valid += 1
            if val_w1 == val_w2:
                w1_w2_matches += 1

        if val_w1 in ["yes", "no"] and val_pred in ["yes", "no"]:
            w1_pred_valid += 1
            if val_w1 == val_pred:
                w1_pred_matches += 1

    w1_w2_acc = w1_w2_matches / w1_w2_valid if w1_w2_valid > 0 else np.nan
    w1_pred_acc = w1_pred_matches / w1_pred_valid if w1_pred_valid > 0 else np.nan
    norm_acc = w1_pred_acc / w1_w2_acc if w1_w2_acc and not np.isnan(w1_pred_acc) else np.nan

    records.append({
        "Email": participant_id,
        "Accuracy_W1_W2": round(w1_w2_acc, 3) if not np.isnan(w1_w2_acc) else np.nan,
        "Accuracy_W1_Pred": round(w1_pred_acc, 3) if not np.isnan(w1_pred_acc) else np.nan,
        "Normalized_Accuracy": round(norm_acc, 3) if not np.isnan(norm_acc) else np.nan
    })

# Save to CSV
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved normalized individual-level accuracy to: {OUTPUT_PATH}")

import pandas as pd
import numpy as np

# Config
WAVE1_CSV     = "jun1wave1_ground_truth.csv"
WAVE2_CSV     = "jun1wave2_ground_truth.csv"
PREDICTED_CSV = "apr8analysispipeline/processed_allcond_responses.csv"
OUTPUT_CSV    = "ALLCOND/adaptation_mitigation_output.csv"

# Mappings
likert_map_7pt = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
    "Unsure": 4, "Neither agree nor disagree": 4, "Somewhat agree": 5,
    "Agree": 6, "Strongly agree": 7
}

def reverse_code(series, max_value):
    return max_value + 1 - series

# Column groups
adaptation_cols = ["Adaptation_1", "Adaptation_2", "Adaptation_3"]
mitigation_cols = [
    "Mitigation_1", "Mitigation_2", "Mitigation_3", "Mitigation_4", "Mitigation_5",
    "Mitigation_6", "Mitigation_7", "Mitigation_8", "Mitigation_9", "Mitigation_10", "Mitigation_11"
]
reverse_mitigation = ["Mitigation_2", "Mitigation_4", "Mitigation_7", "Mitigation_9"]

def load_and_preprocess(csv_path, reverse_cols):
    df = pd.read_csv(csv_path)
    df["Email"] = df["Email"].astype(str).str.lower().str.strip()
    for col in adaptation_cols + mitigation_cols:
        df[col] = df[col].map(likert_map_7pt).astype(float)
    for col in reverse_cols:
        df[col] = reverse_code(df[col], 7)
    return df

def compute_truth_composites(df, wave_suffix):
    # wave_suffix should be "W1" or "W2"
    adaptation_mean = df[adaptation_cols].mean(axis=1)
    mitigation_mean = df[mitigation_cols].mean(axis=1)
    overall_mean = pd.concat([adaptation_mean, mitigation_mean], axis=1).mean(axis=1)
    
    out = pd.DataFrame({
        "Email": df["Email"],
        f"Adaptation_{wave_suffix}": adaptation_mean,
        f"Mitigation_{wave_suffix}": mitigation_mean,
        f"Truth_Overall_{wave_suffix}": overall_mean
    })
    return out

def compute_pred_composites(df):
    adaptation_mean = df[adaptation_cols].mean(axis=1)
    mitigation_mean = df[mitigation_cols].mean(axis=1)
    overall_mean = pd.concat([adaptation_mean, mitigation_mean], axis=1).mean(axis=1)
    
    out = pd.DataFrame({
        "Email": df["Email"],
        "Adaptation_Pred": adaptation_mean,
        "Mitigation_Pred": mitigation_mean,
        "Overall_Pred": overall_mean
    })
    return out

# Load data
df_w1 = load_and_preprocess(WAVE1_CSV, reverse_mitigation)
df_w2 = load_and_preprocess(WAVE2_CSV, reverse_mitigation)
df_pred = load_and_preprocess(PREDICTED_CSV, reverse_mitigation)

# Compute composites
df_w1_comp = compute_truth_composites(df_w1, "W1")
df_w2_comp = compute_truth_composites(df_w2, "W2")
df_pred_comp = compute_pred_composites(df_pred)

# Merge on email
merged = df_w1_comp.merge(df_w2_comp, on="Email").merge(df_pred_comp, on="Email")

# Normalize all composite scores to [0,1] by dividing by 7
for col in merged.columns:
    if col != "Email":
        merged[col] = merged[col] / 7.0

# Save as CSV
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved retest to: {OUTPUT_CSV}")

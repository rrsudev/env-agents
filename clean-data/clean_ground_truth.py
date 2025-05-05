"""
Cleans a raw Qualtrics export to produce a structured ground truth CSV file
for downstream analysis. The script removes metadata rows and retains only 
actual survey responses and validates the question range. 
"""

import pandas as pd

# CONFIG
# path to the original raw CSV file
RAW_CSV_PATH = 'apr8analysispipeline/apr8groundtruth.csv'
# path to save the cleaned CSV file
CLEANED_CSV_PATH = 'apr8analysispipeline/apr8cleangroundtruth.csv'

# PROCESS RAW FILE
df_raw = pd.read_csv(RAW_CSV_PATH, header=None)
# readable column names
new_header = df_raw.iloc[0]

# actual survey response data starts at row index 2:
# row index 0: human-readable column names
# row index 1: Qualtrics metadata 
# row index 2 onward: actual participant responses
df_clean = df_raw.iloc[2:].copy()
df_clean.columns = new_header

# debugging
start_idx = df_clean.columns.get_loc("NEPS_1")
end_idx = df_clean.columns.get_loc("SS_Q6_11")
total_questions = end_idx - start_idx + 1
print(f"\nQuestion range from NEPS_1 to SS_Q6_11: {total_questions} questions.")

# save as CSV
df_clean.to_csv(CLEANED_CSV_PATH, index=False)
print(f"\n✅ Cleaned ground truth CSV saved to: {CLEANED_CSV_PATH}")

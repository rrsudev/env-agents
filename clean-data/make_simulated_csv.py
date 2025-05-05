"""
This parses the JSON-formatted simulated responses from generative agents
and saves a CSV file of all responses sorted by agent. The purpose of the
document is to ease comparisons with ground truth data and allow further 
analysis.
"""

import os
import json
import pandas as pd

# CONFIG
# swap between conditions such as AMERICANVOICES, ENVINTERVIEW, DEMOGRAPHICINTERVIEW
BOVITZ_DIR = 'DEMOGRAPHICINTERVIEW/bovitz'
OUTPUT_CSV = 'apr8analysispipeline/processed_demographic_responses.csv'
MASTER_QUESTION_CSV = 'apr8analysispipeline/susagentmasterquestion.csv'

# QUESTIONS TO INCLUDE
# removed repetitions in questions, Q1–Q232 and Q256–Q385
keep_questions = set(range(1, 233)) | set(range(256, 386)) 
df_master = pd.read_csv(MASTER_QUESTION_CSV)
question_map = dict(zip(df_master.iloc[:, 0], df_master.iloc[:, 1])) 
filtered_map = {q: question_map[q] for q in keep_questions if q in question_map}

# STORE RESPONSES
all_rows = []

# skip folders with unreadable or missing response.json files
for participant_email in os.listdir(BOVITZ_DIR):
    p_path = os.path.join(BOVITZ_DIR, participant_email)
    if not os.path.isdir(p_path):
        continue

    response_map = {}

    for folder in os.listdir(p_path):
        try:
            q_start = int(folder)
        except ValueError:
            continue  # Skip non-numeric folder names

        response_path = os.path.join(p_path, folder, 'response.json')
        if not os.path.exists(response_path):
            print(f"‼️ No valid responses for {participant_email}")
            continue

        try:
            with open(response_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"‼️Error reading {response_path}: {e}")
            continue

        for offset in range(5):
            global_q = q_start + offset
            local_q = str(offset + 1)

            if global_q not in filtered_map:
                continue

            if local_q not in data:
                continue

            entry = data[local_q]
            if not isinstance(entry, dict):
                continue

            response = entry.get("Response")
            if response is None:
                continue

            question_label = filtered_map[global_q]
            response_map[question_label] = str(response).strip()

    if response_map:
        response_map['Email'] = participant_email
        all_rows.append(response_map)

# CSV
columns = ['Email'] + [filtered_map[q] for q in sorted(filtered_map.keys())]
df = pd.DataFrame(all_rows, columns=columns)
df = df.sort_values("Email")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved filtered participant responses to: {OUTPUT_CSV}")

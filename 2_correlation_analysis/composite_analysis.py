"""
====================================================================
 composite_analysis.py
====================================================================

1.  Measure-level correlations (`main`)
    • Walk through each composite-outcome CSV (per measure × agent type).  
    • Compute a *normalized* Pearson correlation  
       (model-vs-W1  ÷  W2-vs-W1).  
    • Save → `composite_outcomes_correlations.csv`.

2.  Person-level correlations (`person_analysis`)
    • Aggregate every participant’s items within a category  
      (`env`, `ind`).  
    • Compute their personal correlation scores for every agent type.  
    • Wide-format output → `composite_outcomes_person_level_correlations.csv`.

---------------------------------------------------------------------
Directory / file assumptions
---------------------------------------------------------------------
Your content should be organized as:

    Composite_Outcomes/
        ├── AMERICANVOICES_waves/
        ├── DEMOGRAPHIC_waves/
        ├── ENVIRONMENTAL_waves/
        └── ALLCONDITIONS_waves/

---------------------------------------------------------------------
Run
---------------------------------------------------------------------
# 1️) Person-level analysis (default) – creates the  CSV
python composite_analysis.py

# 2️) Measure-level correlations
# (uncomment `main()` at bottom or call directly)
python composite_analysis.py  --run-measures

====================================================================
"""


import os
import pandas as pd
from scipy.stats import pearsonr


measures = {
    "adaptation_mitigation_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "Overall_Pred", "w2": "Truth_Overall_W2" , "true": "Truth_Overall_W1"}
        ]
    },
    "cns_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "CNS_Composite_Pred", "w2": "CNS_Composite_W2", "true": "CNS_Composite_W1"}
        ]
    },
    "ecdc_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "Collective_Pred", "w2": "Collective_W2", "true": "Collective_W1"},
            {"pred": "Indiv_Pred", "w2": "Indiv_W2", "true": "Indiv_W1"}
        ]
    },
    "envefficacy_match.csv": {
        "cat": "env",
        "cols": [
            {"pred": "Eff_Indiv_Pred", "w2": "Eff_Indiv_W2", "true": "Eff_Indiv_W1"},
         {"pred": "Eff_Collective_Pred", "w2": "Eff_Collective_W2" , "true": "Eff_Collective_W1"}
        ]
    },
    "envactions_output.csv": {
        "cat": "env",
        "cols": [{"pred": "ENV_ACTIONS_Composite_Pred", "w2":"ENV_ACTIONS_Composite_W2", "true": "ENV_ACTIONS_Composite_W1"}]
    }, 
    "gses_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "GSE_Composite_Pred", "w2": "GSE_Composite_W2", "true": "GSE_Composite_W1"}
        ]
    },
    "gsjs_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "GSJS_Composite_Pred", "w2":  "GSJS_Composite_W2", "true": "GSJS_Composite_W1"}
        ]
    },
    "iri_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "IRI_PT_Pred", "w2": "IRI_PT_W2", "true": "IRI_PT_W1"},
            {"pred": "IRI_FS_Pred", "w2": "IRI_FS_W2", "true": "IRI_FS_W1"},
            {"pred": "IRI_EC_Pred", "w2": "IRI_EC_W2", "true": "IRI_EC_W1"},
            {"pred": "IRI_PD_Pred", "w2": "IRI_PD_W2", "true": "IRI_PD_W1"}
        ]
    },
    "mfq_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "MFQ_overall_Pred", "w2": "MFQ_overall_W2", "true": "MFQ_overall_W1"}
        ]
    },
    "nep_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "NEP_Composite_Pred", "w2": "NEP_Composite_W2" , "true": "NEP_Composite_W1"}
        ]
    },
    "nfc_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "NFC_Composite_Pred", "w2": "NFC_Composite_W2", "true": "NFC_Composite_W1"}
        ]
    },
    "proximity_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "Proximity_Pred", "w2": "Proximity_W2", "true": "Proximity_W1"}
        ]
    },
    "risk_aversion_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "RA_Composite_Pred", "w2": "RA_Composite_W2", "true": "RA_Composite_W1"}
        ]
    },
    "trust_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "Trust_Composite_Pred", "w2": "Trust_Composite_W2", "true": "Trust_Composite_W1"}
        ]
    },
    "sdo_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "SDO_Composite_Pred", "w2":  "SDO_Composite_W2", "true": "SDO_Composite_W1"}
        ]
    },
    "mes_composite_output.csv": {
        "cat": "ind",
        "cols": [
            {"pred": "MES_Composite_Pred", "w2":  "MES_Composite_W2", "true": "MES_Composite_W1"}
        ]
    },
        #"emotions_output.csv": {
        #"cat": "env",
        #"cols": [
        #    {"pred": "Envemotions_Composite_Pred", "w2":  "Envemotions_Composite_W2", "true": "Envemotions_Composite_W1"}
        #]
    #},
}

directories = ["AMERICANVOICES_waves", "DEMOGRAPHIC_waves", "ENVIRONMENTAL_waves", "ALLCONDITIONS_waves"]


def _row_complete(row: pd.Series,
                  pred: str, true: str, w2: str) -> bool:
    return not (pd.isna(row[pred]) or pd.isna(row[true]) or pd.isna(row[w2]))


def get_emails_present_in_every_agent() -> set[str]:
    present_by_agent: dict[str, set[str]] = {d:set() for d in directories}

    for agent_dir in directories:
        agent_ok: set[str] = set()

        for mfile, minfo in measures.items():
            path = os.path.join('Composite_Outcomes', agent_dir, mfile)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)

            for triple in minfo['cols']:
                pred, true, w2 = triple['pred'], triple['true'], triple['w2']
                if not all(c in df.columns for c in (pred, true, w2)):
                    continue

                mask = df.apply(_row_complete, axis=1,
                                args=(pred, true, w2))
                agent_ok.update(df.loc[mask, 'Email'].dropna())

        present_by_agent[agent_dir] = agent_ok

    # keep only participants that appear in every agent condition
    emails_in_all = set.intersection(*present_by_agent.values())
    return emails_in_all



def main():
    results = {}

    # Loop through each agent condition directory
    for directory in directories:
        dir_results = {}

        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping.")
                continue

            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]

            for col_pair in measure_info["cols"]:
                pred_col = col_pair["pred"]
                true_col = col_pair["true"]
                w2_col = col_pair["w2"]

                if pred_col not in df.columns or true_col not in df.columns or w2_col not in df.columns:
                    print(f"Warning: Columns {pred_col} or {true_col} or {w2_col} not found in {path}. Skipping.")
                    continue

                # Drop rows with missing data
                valid_df = df[[pred_col, true_col, w2_col]].dropna()

                if valid_df.shape[0] == 0:
                    print(f"Warning: No valid data for {pred_col} and {true_col} and {w2_col} in {path}. Skipping.")
                    continue

                print("PRED LENGTH: ", len(valid_df[pred_col]))
                print("TRUE LENGTH: ", len(valid_df[true_col]))

                corr, pval = pearsonr(valid_df[pred_col], valid_df[true_col])
                w2_corr, w2_pval = pearsonr(valid_df[w2_col], valid_df[true_col])

                # Key is measure_file + pred/true name
                key = f"{measure_file.replace('.csv', '')}_{pred_col}_vs_{true_col}"
                if key not in results:
                    results[key] = {}

                results[key][f"{directory}_correlation"] = corr/w2_corr
                results[key][f"{directory}_pval"] = pval

    final_df = pd.DataFrame.from_dict(results, orient='index')
    final_df.to_csv("composite_outcomes_correlations.csv")

    print("Done! Output saved to composite_outcomes_correlations.csv")


def person_analysis():
    email_data = {}

    for directory in directories:
        person_cat_data = {}

        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)

            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]


            if 'Email' not in df.columns:
                continue

            cat = measure_info["cat"]

            for _, row in df.iterrows():
                email = row['Email']
                if pd.isna(email):
                    continue

                if email not in person_cat_data:
                    person_cat_data[email] = {}
                if cat not in person_cat_data[email]:
                    person_cat_data[email][cat] = {'pred': [], 'true': [], 'w2': []}

                for col_pair in measure_info["cols"]:
                    pred_col = col_pair["pred"]
                    true_col = col_pair["true"]
                    w2_col = col_pair["w2"]

                    if pred_col not in df.columns or true_col not in df.columns or w2_col not in df.columns:
                        continue

                    pred_val = row[pred_col]
                    true_val = row[true_col]
                    w2_val = row[w2_col]

                    if pd.isna(pred_val) or pd.isna(true_val) or pd.isna(w2_val):
                        continue

                    person_cat_data[email][cat]['pred'].append(pred_val)
                    person_cat_data[email][cat]['true'].append(true_val)
                    person_cat_data[email][cat]['w2'].append(w2_val)

        for email, cat_data in person_cat_data.items():
            if email not in email_data:
                email_data[email] = {}

            for cat, values in cat_data.items():
                preds = values['pred']
                trues = values['true']
                w2s = values['w2']

                if len(preds) < 2 or len(trues) < 2 or len(w2s) < 2:
                    continue

                corr, pval = pearsonr(preds, trues)
                w2_corr, w2_pval = pearsonr(w2s, trues)

                col_corr = f"{directory}_{cat}_correlation"
                col_pval = f"{directory}_{cat}_pval"

                email_data[email][col_corr] = corr
                email_data[email][col_pval] = pval

    final_pivot_df = pd.DataFrame.from_dict(email_data, orient='index').reset_index()
    final_pivot_df = final_pivot_df.rename(columns={'index': 'Email'})
    final_pivot_df.to_csv("composite_outcomes_person_level_correlations.csv", index=False)

    print("Done! Output saved to composite_outcomes_person_level_correlations.csv")


if __name__ == '__main__':
    COMPLETE_EMAILS = get_emails_present_in_every_agent()
    print(f"✓ retaining {len(COMPLETE_EMAILS)} participants present in ALL four agent folders")
    main()
    person_analysis()
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# --------------  CONFIGURATION                          --------------
# ---------------------------------------------------------------------
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
    "emotions_output.csv": {
        "cat": "env",
        "cols": [
            {"pred": "Envemotions_Composite_Pred", "w2":  "Envemotions_Composite_W2", "true": "Envemotions_Composite_W1"}
        ]
    },
}

directories = ["AMERICANVOICES_waves", "DEMOGRAPHIC_waves", "ENVIRONMENTAL_waves", "ALLCONDITIONS_waves"]

# ---------------------------------------------------------------------
# --------------  HELPER FUNCTIONS                       --------------
# ---------------------------------------------------------------------
def fisher_z(r):
    if np.abs(r) == 1.0:
        r = np.sign(r) * (1 - 1e-7)
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return np.tanh(z)

def clamp(r):
    return max(-1.0, min(1.0, r))

def _row_complete(row, pred, true, w2):
    return not (pd.isna(row[pred]) or pd.isna(row[true]) or pd.isna(row[w2]))

def get_emails_present_in_every_agent():
    """
    We identify the participants who have agents properly formed under each condition
    """
    present_by_agent = {d: set() for d in directories}
    for agent_dir in directories:
        agent_ok = set()
        for mfile, minfo in measures.items():
            path = os.path.join('Composite_Outcomes', agent_dir, mfile)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            for triple in minfo['cols']:
                pred, true, w2 = triple['pred'], triple['true'], triple['w2']
                if not all(c in df.columns for c in (pred, true, w2)):
                    continue
                mask = df.apply(_row_complete, axis=1, args=(pred, true, w2))
                agent_ok.update(df.loc[mask, 'Email'].dropna())
        present_by_agent[agent_dir] = agent_ok
    return set.intersection(*present_by_agent.values())

# ---------------------------------------------------------------------
# --------------  MEASURE-LEVEL NORMALIZED CORRELATION   --------------
# ---------------------------------------------------------------------
def main():
    results = {}

    for directory in directories:
        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]

            for col_pair in measure_info["cols"]:
                pred_col, true_col, w2_col = col_pair["pred"], col_pair["true"], col_pair["w2"]
                if not all(c in df.columns for c in (pred_col, true_col, w2_col)):
                    continue

                valid_df = df[[pred_col, true_col, w2_col]].dropna()
                if len(valid_df) < 3:
                    continue

                corr, _ = pearsonr(valid_df[pred_col], valid_df[true_col])
                w2_corr, _ = pearsonr(valid_df[w2_col], valid_df[true_col])

                key = f"{measure_file.replace('.csv','')}_{pred_col}_vs_{true_col}"
                if key not in results:
                    results[key] = {}

                if pd.isna(corr) or pd.isna(w2_corr) or abs(w2_corr) < 1e-6:
                    continue
                norm_corr = corr/w2_corr

                results[key][f"{directory}_correlation"] = norm_corr
                results[key][f"{directory}_raw_r"] = corr
                results[key][f"{directory}_w2_r"] = w2_corr

    pd.DataFrame.from_dict(results, orient='index').to_csv("composite_outcomes_correlations.csv")
    print("✓ composite_outcomes_correlations.csv written")


# ---------------------------------------------------------------------
# --------------  PERSON-LEVEL NORMALIZED CORRELATION    --------------
# ---------------------------------------------------------------------
def person_analysis():
    email_data = {}

    for directory in directories:
        person_cat_data = {}

        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)
            if not os.path.exists(path): continue

            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]

            for _, row in df.iterrows():
                email = row.get("Email")
                if pd.isna(email): continue

                cat = measure_info["cat"]
                if email not in person_cat_data:
                    person_cat_data[email] = {}
                if cat not in person_cat_data[email]:
                    person_cat_data[email][cat] = {"pred": [], "true": [], "w2": []}

                for col_pair in measure_info["cols"]:
                    pred, true, w2 = col_pair["pred"], col_pair["true"], col_pair["w2"]
                    if any(c not in df.columns for c in (pred, true, w2)):
                        continue
                    if pd.isna(row[pred]) or pd.isna(row[true]) or pd.isna(row[w2]):
                        continue

                    person_cat_data[email][cat]["pred"].append(row[pred])
                    person_cat_data[email][cat]["true"].append(row[true])
                    person_cat_data[email][cat]["w2"].append(row[w2])

        for email, cats in person_cat_data.items():
            if email not in email_data:
                email_data[email] = {}

            for cat, data in cats.items():
                preds = data["pred"]
                trues = data["true"]
                w2s = data["w2"]

                if len(preds) < 2 or len(trues) < 2 or len(w2s) < 2:
                    continue

                corr, _ = pearsonr(preds, trues)
                w2_corr, _ = pearsonr(w2s, trues)

                col_corr = f"{directory}_{cat}_correlation"
                col_norm = f"{directory}_{cat}_normalized"

                email_data[email][col_corr] = corr
                if pd.isna(corr) or pd.isna(w2_corr) or abs(w2_corr) < 1e-6:
                    continue
                norm_corr = corr/w2_corr

                email_data[email][col_norm] = norm_corr

    df = pd.DataFrame.from_dict(email_data, orient="index").reset_index().rename(columns={"index": "Email"})
    df.to_csv("composite_outcomes_person_level_correlations.csv", index=False)
    print("✓ composite_outcomes_person_level_correlations.csv written")


# ---------------------------------------------------------------------
# --------------  MEASURE-LEVEL RAW CORRELATIONS         --------------
# ---------------------------------------------------------------------


def measure_raw():
    rows = []

    for directory in directories:
        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]

            for col_pair in measure_info["cols"]:
                pred_col, true_col, w2_col = col_pair["pred"], col_pair["true"], col_pair["w2"]
                if not all(c in df.columns for c in (pred_col, true_col, w2_col)):
                    continue

                valid_df = df[[pred_col, true_col, w2_col]].dropna()
                if len(valid_df) < 3:
                    continue

                r_pred, _ = pearsonr(valid_df[pred_col], valid_df[true_col])
                r_w2, _ = pearsonr(valid_df[w2_col], valid_df[true_col])

                rows.append({
                    "agent": directory,
                    "measure_file": measure_file,
                    "pred_col": pred_col,
                    "true_col": true_col,
                    "w2_col": w2_col,
                    "r_pred_clamped": clamp(r_pred),
                    "r_w2_clamped": clamp(r_w2),
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("composite_outcomes_measure_raw_correlations.csv", index=False)
    print("✓ composite_outcomes_measure_raw_correlations.csv written")



# ---------------------------------------------------------------------
# --------------  PERSON-LEVEL RAW CORRELATIONS          --------------
# ---------------------------------------------------------------------
def person_raw():
    email_data = []

    for directory in directories:
        person_cat_data = {}

        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)
            if not os.path.exists(path): continue

            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]

            for _, row in df.iterrows():
                email = row.get("Email")
                if pd.isna(email): continue

                cat = measure_info["cat"]
                if email not in person_cat_data:
                    person_cat_data[email] = {}
                if cat not in person_cat_data[email]:
                    person_cat_data[email][cat] = {"pred": [], "true": [], "w2": []}

                for col_pair in measure_info["cols"]:
                    pred, true, w2 = col_pair["pred"], col_pair["true"], col_pair["w2"]
                    if any(c not in df.columns for c in (pred, true, w2)):
                        continue
                    if pd.isna(row[pred]) or pd.isna(row[true]) or pd.isna(row[w2]):
                        continue

                    person_cat_data[email][cat]["pred"].append(row[pred])
                    person_cat_data[email][cat]["true"].append(row[true])
                    person_cat_data[email][cat]["w2"].append(row[w2])

        for email, cats in person_cat_data.items():
            for cat, data in cats.items():
                preds = data["pred"]
                trues = data["true"]
                w2s = data["w2"]

                if len(preds) < 2 or len(trues) < 2 or len(w2s) < 2:
                    continue

                r_pred, _ = pearsonr(preds, trues)
                r_w2, _ = pearsonr(w2s, trues)

                email_data.append({
                    "Email": email,
                    "agent": directory,
                    "category": cat,
                    "r_pred_clamped": clamp(r_pred),
                    "r_w2_clamped": clamp(r_w2)
                })

    df = pd.DataFrame(email_data)
    df.to_csv("composite_outcomes_person_raw_correlations.csv", index=False)
    print("✓ composite_outcomes_person_raw_correlations.csv written")


# ---------------------------------------------------------------------
# --------------  DIAGNOSTICS FOR COMPOSITE OUTCOMES     --------------
# ---------------------------------------------------------------------
def diagnostic_report():
    rows = []

    for directory in directories:
        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = df[df['Email'].isin(COMPLETE_EMAILS)]

            for col_pair in measure_info["cols"]:
                pred_col = col_pair["pred"]
                true_col = col_pair["true"]
                w2_col = col_pair["w2"]

                if not all(c in df.columns for c in (pred_col, true_col, w2_col)):
                    continue

                valid_df = df[[pred_col, true_col, w2_col]].dropna()
                if len(valid_df) < 3:
                    continue

                r_pred, _ = pearsonr(valid_df[pred_col], valid_df[true_col])
                r_w2, _ = pearsonr(valid_df[w2_col], valid_df[true_col])

                if np.isnan(r_pred) or np.isnan(r_w2):
                    continue

                try:
                    z_pred = fisher_z(r_pred)
                    z_w2 = fisher_z(r_w2)
                    z_diff = z_pred - z_w2
                except Exception:
                    z_diff = np.nan

                try:
                    norm_ratio = r_pred / r_w2 if r_w2 != 0 else np.nan
                except ZeroDivisionError:
                    norm_ratio = np.nan

                rows.append({
                    "agent": directory,
                    "measure_file": measure_file,
                    "pred_col": pred_col,
                    "true_col": true_col,
                    "w2_col": w2_col,
                    "r_pred": r_pred,
                    "r_w2": r_w2,
                    "z_subtraction": z_diff,
                    "normalized_correlation_ratio": norm_ratio
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("composite_outcomes_raw_correlations_diagnostics.csv", index=False)
    print("✓ composite_outcomes_raw_correlations_diagnostics.csv written")



# -----------------------------------------------------------------
# --------------  DISPATCHER       --------------------------------
# -----------------------------------------------------------------
if __name__ == "__main__":
    COMPLETE_EMAILS = get_emails_present_in_every_agent()
    print(f"✓ retaining {len(COMPLETE_EMAILS)} participants present in ALL four agent folders")

    main()
    person_analysis()
    measure_raw()
    person_raw()
    # diagnostic_report()



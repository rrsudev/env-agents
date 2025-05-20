import os
import pandas as pd
from scipy.stats import pearsonr

# measures = {"adaptation_mitigation_output.csv": {"cat": "env", "cols": [
#     {"pred": "Overall_Composite_Pred", "true": "Overall_Composite_Truth"}]},
#             "cns_output.csv": {"cat": "env",
#                                "cols": [{"pred": "CNS_Composite_Pred_Norm", "true": "CNS_Composite_Truth_Norm"}]},
#             "ecdc_output.csv": {"cat": "env", "cols": [{"pred": "ECDC_Collect_Pred", "true": "ECDC_Collect_Truth"},
#                                                        {"pred": "ECDC_Indiv_Pred", "true": "ECDC_Indiv_Truth"}]},
#             "efficacy_composite_match.csv": {"cat": "env",
#                                              "cols": [{"pred": "Eff_Indiv_pred", "true": "Eff_Indiv_truth"},
#                                                       {"pred": "Eff_Collective_pred", "true": "Eff_Collective_truth"}]},
#             # "envactions_output.csv": {"cat": "env", "cols": [{"pred": "ENV_Actions_Composite_Pred", "true": "ENV_Actions_Composite_Truth"}]}, # not a scale or sum
#             # "envemotions_output.csv": {"cat": "env", "cols": [{"pred": , "true": }]}, # This one is just match rate on a list of emotions
#             "gse_individual_level_summary.csv": {"cat": "ind", "cols": [
#                 {"pred": "GSE_Composite_Pred", "true": "GSE_Composite_Truth"}]},
#             "gsjs_output.csv": {"cat": "soc",
#                                 "cols": [{"pred": "GSJS_Composite_Pred", "true": "GSJS_Composite_Truth"}]},
#             "iri_output.csv": {"cat": "ind", "cols": [{"pred": "IRI_PT_pred", "true": "IRI_PT_truth"},
#                                                       {"pred": "IRI_FS_pred", "true": "IRI_FS_truth"},
#                                                       {"pred": "IRI_EC_pred", "true": "IRI_EC_truth"},
#                                                       {"pred": "IRI_PD_pred", "true": "IRI_PD_truth"}]},
#             "mfq_output.csv": {"cat": "soc", "cols": [{"pred": "MFQ_overall_pred", "true": "MFQ_overall_truth"}]},
#             "nep_output.csv": {"cat": "env", "cols": [{"pred": "NEP_Composite_Pred", "true": "NEP_Composite_Truth"}]},
#             "nfc_output.csv": {"cat": "ind", "cols": [{"pred": "NFC_Composite_Pred", "true": "NFC_Composite_Truth"}]},
#             "proximity_output.csv": {"cat": "env", "cols": [
#                 {"pred": "Proximity_Composite_Pred", "true": "Proximity_Composite_Truth"}]},
#             "risk_aversion_output.csv": {"cat": "ind",
#                                          "cols": [{"pred": "Switch_Point_Pred", "true": "Switch_Point_Truth"}]},
#             "sciencerank_normalized_output.csv": {"cat": "ind", "cols": [
#                 {"pred": "Science_Rank_Overall_Pred", "true": "Science_Rank_Overall_Truth"}]},
#             "sdo_output.csv": {"cat": "soc", "cols": [{"pred": "SDO_Composite_Pred", "true": "SDO_Composite_Truth"}]},
#             # "svi_output.csv": {"cat": "ind", "cols": [{"pred": , "true": }]} # separate not composite
#             }
#
# all_measures = {"adaptation_mitigation_output.csv": {"cat": "env", "cols": [
#     {"pred": "Overall_Composite_Pred", "true": "Overall_Composite_Truth"}]},
#                 "cns_output.csv": {"cat": "env",
#                                    "cols": [{"pred": "CNS_Composite_Pred_Norm", "true": "CNS_Composite_Truth_Norm"}]},
#                 "ecdc_output.csv": {"cat": "env", "cols": [{"pred": "ECDC_Collect_Pred", "true": "ECDC_Collect_Truth"},
#                                                            {"pred": "ECDC_Indiv_Pred", "true": "ECDC_Indiv_Truth"}]},
#                 "efficacy_composite_match.csv": {"cat": "env",
#                                                  "cols": [{"pred": "Eff_Indiv_pred", "true": "Eff_Indiv_truth"},
#                                                           {"pred": "Eff_Collective_pred",
#                                                            "true": "Eff_Collective_truth"}]},
#                 "envactions_output.csv": {"cat": "env", "cols": [
#     {"pred": "ENV_ACTIONS_1_pred", "true": "ENV_ACTIONS_1_truth"},
#     {"pred": "ENV_ACTIONS_2_pred", "true": "ENV_ACTIONS_2_truth"},
#     {"pred": "ENV_ACTIONS_3_pred", "true": "ENV_ACTIONS_3_truth"},
#     {"pred": "ENV_ACTIONS_4_pred", "true": "ENV_ACTIONS_4_truth"},
#     {"pred": "ENV_ACTIONS_5_pred", "true": "ENV_ACTIONS_5_truth"},
#     {"pred": "ENV_ACTIONS_6_pred", "true": "ENV_ACTIONS_6_truth"},
#     {"pred": "ENV_ACTIONS_7_pred", "true": "ENV_ACTIONS_7_truth"},
#     {"pred": "ENV_ACTIONS_8_pred", "true": "ENV_ACTIONS_8_truth"},
#     {"pred": "ENV_ACTIONS_9_pred", "true": "ENV_ACTIONS_9_truth"},
#     {"pred": "ENV_ACTIONS_10_pred", "true": "ENV_ACTIONS_10_truth"},
#     {"pred": "ENV_ACTIONS_11_pred", "true": "ENV_ACTIONS_11_truth"},
#     {"pred": "ENV_ACTIONS_12_pred", "true": "ENV_ACTIONS_12_truth"},
#     {"pred": "ENV_ACTIONS_13_pred", "true": "ENV_ACTIONS_13_truth"},
#     {"pred": "ENV_ACTIONS_14_pred", "true": "ENV_ACTIONS_14_truth"},
#     {"pred": "ENV_ACTIONS_15_pred", "true": "ENV_ACTIONS_15_truth"},
#     {"pred": "ENV_ACTIONS_16_pred", "true": "ENV_ACTIONS_16_truth"},
#     {"pred": "ENV_ACTIONS_17_pred", "true": "ENV_ACTIONS_17_truth"}]
# },
#                 # not a scale or sum
#                 "envemotions_output.csv": {"cat": "env", "cols": [{"matchRate": "Emotion_Direct_Match_Rate"}]},
#                 # This one is just match rate on a list of emotions
#                 "gse_individual_level_summary.csv": {"cat": "ind", "cols": [
#                     {"pred": "GSE_Composite_Pred", "true": "GSE_Composite_Truth"}]},
#                 "gsjs_output.csv": {"cat": "soc",
#                                     "cols": [{"pred": "GSJS_Composite_Pred", "true": "GSJS_Composite_Truth"}]},
#                 "iri_output.csv": {"cat": "ind", "cols": [{"pred": "IRI_PT_pred", "true": "IRI_PT_truth"},
#                                                           {"pred": "IRI_FS_pred", "true": "IRI_FS_truth"},
#                                                           {"pred": "IRI_EC_pred", "true": "IRI_EC_truth"},
#                                                           {"pred": "IRI_PD_pred", "true": "IRI_PD_truth"}]},
#                 "mfq_output.csv": {"cat": "soc", "cols": [{"pred": "MFQ_overall_pred", "true": "MFQ_overall_truth"}]},
#                 "nep_output.csv": {"cat": "env",
#                                    "cols": [{"pred": "NEP_Composite_Pred", "true": "NEP_Composite_Truth"}]},
#                 "nfc_output.csv": {"cat": "ind",
#                                    "cols": [{"pred": "NFC_Composite_Pred", "true": "NFC_Composite_Truth"}]},
#                 "proximity_output.csv": {"cat": "env", "cols": [
#                     {"pred": "Proximity_Composite_Pred", "true": "Proximity_Composite_Truth"}]},
#                 "risk_aversion_output.csv": {"cat": "ind",
#                                              "cols": [{"pred": "Switch_Point_Pred", "true": "Switch_Point_Truth"}]},
#                 "sciencerank_normalized_output.csv": {"cat": "ind", "cols": [
#                     {"pred": "Science_Rank_Overall_Pred", "true": "Science_Rank_Overall_Truth"}]},
#                 "sdo_output.csv": {"cat": "soc",
#                                    "cols": [{"pred": "SDO_Composite_Pred", "true": "SDO_Composite_Truth"}]},
#                 "svi_output.csv": {"cat": "ind",
#                                    "cols": [{"pred": "Exact_Match_Rate", "true": "Directional_Match_Rate"}]}
#                 # separate not composite
#                 }

measures = {"adaptation_mitigation_output.csv": {"cat": "env", "cols": [
    {"pred": "Overall_Composite_Pred", "true": "Overall_Composite_Truth"}]},
            "cns_output.csv": {"cat": "env",
                               "cols": [{"pred": "CNS_Composite_Pred_Norm", "true": "CNS_Composite_Truth_Norm"}]},
            "ecdc_output.csv": {"cat": "env", "cols": [{"pred": "ECDC_Collect_Pred", "true": "ECDC_Collect_Truth"},
                                                       {"pred": "ECDC_Indiv_Pred", "true": "ECDC_Indiv_Truth"}]},
            "efficacy_composite_match.csv": {"cat": "env",
                                             "cols": [{"pred": "Eff_Indiv_pred", "true": "Eff_Indiv_truth"},
                                                      {"pred": "Eff_Collective_pred", "true": "Eff_Collective_truth"}]},
            # "envactions_output.csv": {"cat": "env", "cols": [{"pred": "ENV_Actions_Composite_Pred", "true": "ENV_Actions_Composite_Truth"}]}, # not a scale or sum
            # "envemotions_output.csv": {"cat": "env", "cols": [{"pred": , "true": }]}, # This one is just match rate on a list of emotions
            "gse_individual_level_summary.csv": {"cat": "ind", "cols": [
                {"pred": "GSE_Composite_Pred", "true": "GSE_Composite_Truth"}]},
            "gsjs_output.csv": {"cat": "ind",
                                "cols": [{"pred": "GSJS_Composite_Pred", "true": "GSJS_Composite_Truth"}]},
            "iri_output.csv": {"cat": "ind", "cols": [{"pred": "IRI_PT_pred", "true": "IRI_PT_truth"},
                                                      {"pred": "IRI_FS_pred", "true": "IRI_FS_truth"},
                                                      {"pred": "IRI_EC_pred", "true": "IRI_EC_truth"},
                                                      {"pred": "IRI_PD_pred", "true": "IRI_PD_truth"}]},
            "mfq_output.csv": {"cat": "ind", "cols": [{"pred": "MFQ_overall_pred", "true": "MFQ_overall_truth"}]},
            "nep_output.csv": {"cat": "env", "cols": [{"pred": "NEP_Composite_Pred", "true": "NEP_Composite_Truth"}]},
            "nfc_output.csv": {"cat": "ind", "cols": [{"pred": "NFC_Composite_Pred", "true": "NFC_Composite_Truth"}]},
            "proximity_output.csv": {"cat": "env", "cols": [
                {"pred": "Proximity_Composite_Pred", "true": "Proximity_Composite_Truth"}]},
            "risk_aversion_output.csv": {"cat": "ind",
                                         "cols": [{"pred": "Switch_Point_Pred", "true": "Switch_Point_Truth"}]},
            "sciencerank_normalized_output.csv": {"cat": "ind", "cols": [
                {"pred": "Science_Rank_Overall_Pred", "true": "Science_Rank_Overall_Truth"}]},
            "sdo_output.csv": {"cat": "ind", "cols": [{"pred": "SDO_Composite_Pred", "true": "SDO_Composite_Truth"}]},
            # "svi_output.csv": {"cat": "ind", "cols": [{"pred": , "true": }]} # separate not composite
            }

all_measures = {"adaptation_mitigation_output.csv": {"cat": "env", "cols": [
    {"pred": "Overall_Composite_Pred", "true": "Overall_Composite_Truth"}]},
                "cns_output.csv": {"cat": "env",
                                   "cols": [{"pred": "CNS_Composite_Pred_Norm", "true": "CNS_Composite_Truth_Norm"}]},
                "ecdc_output.csv": {"cat": "env", "cols": [{"pred": "ECDC_Collect_Pred", "true": "ECDC_Collect_Truth"},
                                                           {"pred": "ECDC_Indiv_Pred", "true": "ECDC_Indiv_Truth"}]},
                "efficacy_composite_match.csv": {"cat": "env",
                                                 "cols": [{"pred": "Eff_Indiv_pred", "true": "Eff_Indiv_truth"},
                                                          {"pred": "Eff_Collective_pred",
                                                           "true": "Eff_Collective_truth"}]},
                "envactions_output.csv": {"cat": "env", "cols": [
    {"pred": "ENV_ACTIONS_1_pred", "true": "ENV_ACTIONS_1_truth"},
    {"pred": "ENV_ACTIONS_2_pred", "true": "ENV_ACTIONS_2_truth"},
    {"pred": "ENV_ACTIONS_3_pred", "true": "ENV_ACTIONS_3_truth"},
    {"pred": "ENV_ACTIONS_4_pred", "true": "ENV_ACTIONS_4_truth"},
    {"pred": "ENV_ACTIONS_5_pred", "true": "ENV_ACTIONS_5_truth"},
    {"pred": "ENV_ACTIONS_6_pred", "true": "ENV_ACTIONS_6_truth"},
    {"pred": "ENV_ACTIONS_7_pred", "true": "ENV_ACTIONS_7_truth"},
    {"pred": "ENV_ACTIONS_8_pred", "true": "ENV_ACTIONS_8_truth"},
    {"pred": "ENV_ACTIONS_9_pred", "true": "ENV_ACTIONS_9_truth"},
    {"pred": "ENV_ACTIONS_10_pred", "true": "ENV_ACTIONS_10_truth"},
    {"pred": "ENV_ACTIONS_11_pred", "true": "ENV_ACTIONS_11_truth"},
    {"pred": "ENV_ACTIONS_12_pred", "true": "ENV_ACTIONS_12_truth"},
    {"pred": "ENV_ACTIONS_13_pred", "true": "ENV_ACTIONS_13_truth"},
    {"pred": "ENV_ACTIONS_14_pred", "true": "ENV_ACTIONS_14_truth"},
    {"pred": "ENV_ACTIONS_15_pred", "true": "ENV_ACTIONS_15_truth"},
    {"pred": "ENV_ACTIONS_16_pred", "true": "ENV_ACTIONS_16_truth"},
    {"pred": "ENV_ACTIONS_17_pred", "true": "ENV_ACTIONS_17_truth"}]
},
                # not a scale or sum
                "envemotions_output.csv": {"cat": "env", "cols": [{"matchRate": "Emotion_Direct_Match_Rate"}]},
                # This one is just match rate on a list of emotions
                "gse_individual_level_summary.csv": {"cat": "ind", "cols": [
                    {"pred": "GSE_Composite_Pred", "true": "GSE_Composite_Truth"}]},
                "gsjs_output.csv": {"cat": "ind",
                                    "cols": [{"pred": "GSJS_Composite_Pred", "true": "GSJS_Composite_Truth"}]},
                "iri_output.csv": {"cat": "ind", "cols": [{"pred": "IRI_PT_pred", "true": "IRI_PT_truth"},
                                                          {"pred": "IRI_FS_pred", "true": "IRI_FS_truth"},
                                                          {"pred": "IRI_EC_pred", "true": "IRI_EC_truth"},
                                                          {"pred": "IRI_PD_pred", "true": "IRI_PD_truth"}]},
                "mfq_output.csv": {"cat": "ind", "cols": [{"pred": "MFQ_overall_pred", "true": "MFQ_overall_truth"}]},
                "nep_output.csv": {"cat": "env",
                                   "cols": [{"pred": "NEP_Composite_Pred", "true": "NEP_Composite_Truth"}]},
                "nfc_output.csv": {"cat": "ind",
                                   "cols": [{"pred": "NFC_Composite_Pred", "true": "NFC_Composite_Truth"}]},
                "proximity_output.csv": {"cat": "env", "cols": [
                    {"pred": "Proximity_Composite_Pred", "true": "Proximity_Composite_Truth"}]},
                "risk_aversion_output.csv": {"cat": "ind",
                                             "cols": [{"pred": "Switch_Point_Pred", "true": "Switch_Point_Truth"}]},
                "sciencerank_normalized_output.csv": {"cat": "ind", "cols": [
                    {"pred": "Science_Rank_Overall_Pred", "true": "Science_Rank_Overall_Truth"}]},
                "sdo_output.csv": {"cat": "ind",
                                   "cols": [{"pred": "SDO_Composite_Pred", "true": "SDO_Composite_Truth"}]},
                "svi_output.csv": {"cat": "ind",
                                   "cols": [{"pred": "Exact_Match_Rate", "true": "Directional_Match_Rate"}]}
                # separate not composite
                }


# rename emotion_direct_match_rate to envemotions
# rename svi rates

directories = ["AVPaggregated_outputs", "DEMOGRAPHICaggregated_outputs", "ENVIRONMENTALaggregated_outputs"]


def main():
    results = {}

    # Loop through each directory
    for directory in directories:
        dir_results = {}

        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping.")
                continue

            df = pd.read_csv(path)

            for col_pair in measure_info["cols"]:
                pred_col = col_pair["pred"]
                true_col = col_pair["true"]

                if pred_col not in df.columns or true_col not in df.columns:
                    print(f"Warning: Columns {pred_col} or {true_col} not found in {path}. Skipping.")
                    continue

                # Drop rows with missing data
                valid_df = df[[pred_col, true_col]].dropna()

                if valid_df.shape[0] == 0:
                    print(f"Warning: No valid data for {pred_col} and {true_col} in {path}. Skipping.")
                    continue

                corr, pval = pearsonr(valid_df[pred_col], valid_df[true_col])

                # Key is measure_file + pred/true name
                key = f"{measure_file.replace('.csv', '')}_{pred_col}_vs_{true_col}"
                if key not in results:
                    results[key] = {}

                results[key][f"{directory}_correlation"] = corr
                results[key][f"{directory}_pval"] = pval

    # Convert to DataFrame
    final_df = pd.DataFrame.from_dict(results, orient='index')

    # Save
    final_df.to_csv("composite_outcomes_correlations.csv")

    print("Done! Output saved to composite_outcomes_correlations.csv")


def person_analysis():
    # This will collect results like: {Email: {directory_cat_correlation: value, directory_cat_pval: value, ...}}
    email_data = {}

    # Loop through each directory
    for directory in directories:
        # Nested container: {Email -> {cat -> lists of (pred, true) pairs}}
        person_cat_data = {}

        for measure_file, measure_info in measures.items():
            path = os.path.join('Composite_Outcomes', directory, measure_file)

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping.")
                continue

            df = pd.read_csv(path)

            if 'Email' not in df.columns:
                print(f"Warning: 'Email' column missing in {path}. Skipping.")
                continue

            cat = measure_info["cat"]

            for _, row in df.iterrows():
                email = row['Email']
                if pd.isna(email):
                    continue  # skip if no email

                if email not in person_cat_data:
                    person_cat_data[email] = {}
                if cat not in person_cat_data[email]:
                    person_cat_data[email][cat] = {'pred': [], 'true': []}

                for col_pair in measure_info["cols"]:
                    pred_col = col_pair["pred"]
                    true_col = col_pair["true"]

                    if pred_col not in df.columns or true_col not in df.columns:
                        continue

                    pred_val = row[pred_col]
                    true_val = row[true_col]

                    if pd.isna(pred_val) or pd.isna(true_val):
                        continue  # skip missing values

                    person_cat_data[email][cat]['pred'].append(pred_val)
                    person_cat_data[email][cat]['true'].append(true_val)

        # Now for each Email-cat pair, compute the correlation
        for email, cat_data in person_cat_data.items():
            if email not in email_data:
                email_data[email] = {}

            for cat, values in cat_data.items():
                preds = values['pred']
                trues = values['true']

                if len(preds) < 2:
                    # Need at least 2 points for correlation
                    corr, pval = (float('nan'), float('nan'))
                else:
                    corr, pval = pearsonr(preds, trues)

                col_corr = f"{directory}_{cat}_correlation"
                col_pval = f"{directory}_{cat}_pval"

                email_data[email][col_corr] = corr
                email_data[email][col_pval] = pval

    # Now turn the big email_data dict into a DataFrame
    final_pivot_df = pd.DataFrame.from_dict(email_data, orient='index').reset_index()
    final_pivot_df = final_pivot_df.rename(columns={'index': 'Email'})

    # Save
    final_pivot_df.to_csv("composite_outcomes_person_level_correlations_pivoted.csv", index=False)

    print("Done! Output saved to composite_outcomes_person_level_correlations_pivoted.csv")


def compile_all_composites_one_csv():
    # Step 1: Collect all unique emails to use as the base
    all_emails = set()

    for dir_ in directories:
        for filename in all_measures:
            path = os.path.join("Composite_Outcomes", dir_, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if "Email" in df.columns:
                    all_emails.update(df["Email"].dropna().unique())

    print("all emails size: ", len(all_emails))
    # Start with base DataFrame of all emails
    merged_df = pd.DataFrame({"Email": list(all_emails)})

    # Step 2: Iteratively merge desired columns
    for dir_ in directories:
        for filename, info in all_measures.items():
            path = os.path.join("Composite_Outcomes", dir_, filename)
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            if "Email" not in df.columns:
                continue

            cols_to_add = ["Email"]
            for col_group in info["cols"]:
                for key, colname in col_group.items():
                    new_colname = f"{dir_}_{filename}_{colname}"
                    df.rename(columns={colname: new_colname}, inplace=True)
                    cols_to_add.append(new_colname)

            df = df[cols_to_add]

            # Important step: Reduce to one row per email
            df = df.groupby("Email", as_index=False).first()

            merged_df = pd.merge(merged_df, df, on="Email", how="left")

            print(f"{path}\nSize after merge: {merged_df.shape}")

    merged_df.to_csv("compiled_output.csv", index=False)


if __name__ == '__main__':
    person_analysis()
    main()
    #compile_all_composites_one_csv()

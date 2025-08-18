import os
import pandas as pd
from scipy.stats import pearsonr

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------------------------------------------------------------------
# GLOBAL CONFIG 
# ---------------------------------------------------------------------
# !!! TO THE USER: Make sure these are accurate to your own file structure
DATA_W1         = "groundtruth_data/w1_groundtruth.csv"
DATA_W2         = "groundtruth_data/w2_groundtruth.csv"
DATA_PRED       = "simulated_data/simulated_allcond.csv",

# pathways for agent conditions
# !!! TO THE USER: Make sure these are accurate to your own file structure
PRED_FILES = {
    "ALLCONDITIONS":        "simulated_data/simulated_allcond.csv",
    "AMERICANVOICES":       "simulated_data/simulated_americanvoices.csv",
    "DEMOGRAPHIC":          "simulated_data/simulated_demographic.csv",
    "ENVIRONMENTAL":        "simulated_data/simulated_environment.csv",
}

def run_two_way_anova_from_person_level(
        csv_path: str = 'composite_outcomes_person_level_correlations.csv',
        alpha: float = 0.05,
        verbose: bool = True
    ):
    """
    Reads the wide  CSV, converts to long format, runs a two-way ANOVA
    on correlation ~ agent_type * measure_type, and executes Tukey HSD
    comparisons (overall and within each measure_type).
    """
      
    # --------------------------- 1. Load ----------------------------
    df_wide = pd.read_csv(csv_path)

    agent_types = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    measure_types = ['env', 'ind']

    records = []
    for agent in agent_types:
        for measure in measure_types:
            col = f"{agent}_waves_{measure}_correlation"
            if col not in df_wide.columns:
                continue
            for val in df_wide[col].dropna():
                records.append(
                    {'correlation': val,
                     'agent_type' : agent,
                     'measure_type': measure}
                )

    long_df = pd.DataFrame.from_records(records)

    # --------------------------- 2. ANOVA ----------------------------------
    model  = smf.ols('correlation ~ C(agent_type) * C(measure_type)',
                     data=long_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # --------------------------- 3. Tukey HSD ------------------------------
    # 3-a. Overall (agent_type, pooling across measure_type)
    tukey_overall = pairwise_tukeyhsd(
        endog=long_df['correlation'],
        groups=long_df['agent_type'],
        alpha=alpha
    )

    # 3-b. Within each measure_type
    tukey_by_measure = {}
    for mtype in long_df['measure_type'].unique():
        subset = long_df[long_df['measure_type'] == mtype]
        if subset['agent_type'].nunique() < 2:
            # need at least two groups to compare
            continue
        tukey_by_measure[mtype] = pairwise_tukeyhsd(
            endog=subset['correlation'],
            groups=subset['agent_type'],
            alpha=alpha
        )

    # --------------------------- 4. Verbose output -------------------------
    if verbose:
        print("\n--- Two-way ANOVA (Type II) ---")
        print(anova_table)

        print("\n--- Tukey HSD: agent_type (pooled across measure_type) ---")
        print(tukey_overall.summary())

        for mtype, tuk in tukey_by_measure.items():
            print(f"\n--- Tukey HSD: agent_type within measure_type = {mtype} ---")
            print(tuk.summary())

    return {
        'long_df'        : long_df,
        'anova'          : anova_table,
        'tukey_overall'  : tukey_overall,
        'tukey_by_measure': tukey_by_measure
    }


def run_two_way_anova_from_measure_level(
        csv_path: str = 'composite_outcomes_correlations.csv',
        alpha: float = 0.05,
        verbose: bool = True
    ):
    """
    Reads the measure-level CSV, converts to long format, runs a two-way ANOVA
    on correlation ~ agent_type * measure_type, and executes Tukey HSD
    comparisons (overall and within each measure_type).
    """
      
    # --------------------------- 1. Load ----------------------------
    df_wide = pd.read_csv(csv_path, index_col=0)

    # Define the measures configuration (same as in plot_composite_analysis.py)
    measures = {
        "adaptation_mitigation_output.csv": {"cat": "env"},
        "cns_output.csv": {"cat": "env"},
        "envefficacy_match.csv": {"cat": "env"},
        "envactions_output.csv": {"cat": "env"},
        "gses_output.csv": {"cat": "ind"},
        "gsjs_output.csv": {"cat": "ind"},
        "iri_output.csv": {"cat": "ind"},
        "mfq_output.csv": {"cat": "ind"},
        "nep_output.csv": {"cat": "env"},
        "nfc_output.csv": {"cat": "ind"},
        "proximity_output.csv": {"cat": "env"},
        "risk_aversion_output.csv": {"cat": "ind"},
        "trust_output.csv": {"cat": "env"},
        "sdo_output.csv": {"cat": "ind"},
        "mes_composite_output.csv": {"cat": "ind"},
        "emotions_output.csv": {"cat": "env"}
    }

    agent_types = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']

    records = []
    for index in df_wide.index:
        # Extract measure name from index (e.g., "adaptation_mitigation_output_Overall_Pred_vs_Truth_Overall_W1")
        # Find which measure this corresponds to
        measure_type = None
        for measure_file, info in measures.items():
            if measure_file.replace('.csv', '') in index:
                measure_type = info['cat']
                break
        
        if measure_type is None:
            continue  # Skip if we can't determine the measure type
            
        for agent in agent_types:
            col = f"{agent}_waves_correlation"
            if col in df_wide.columns:
                val = df_wide.loc[index, col]
                if not pd.isna(val):
                    records.append({
                        'correlation': val,
                        'agent_type': agent,
                        'measure_type': measure_type
                    })

    long_df = pd.DataFrame.from_records(records)

    # --------------------------- 2. ANOVA ----------------------------------
    model = smf.ols('correlation ~ C(agent_type) * C(measure_type)',
                     data=long_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # --------------------------- 3. Tukey HSD ------------------------------
    # 3-a. Overall (agent_type, pooling across measure_type)
    tukey_overall = pairwise_tukeyhsd(
        endog=long_df['correlation'],
        groups=long_df['agent_type'],
        alpha=alpha
    )

    # 3-b. Within each measure_type
    tukey_by_measure = {}
    for mtype in long_df['measure_type'].unique():
        subset = long_df[long_df['measure_type'] == mtype]
        if subset['agent_type'].nunique() < 2:
            # need at least two groups to compare
            continue
        tukey_by_measure[mtype] = pairwise_tukeyhsd(
            endog=subset['correlation'],
            groups=subset['agent_type'],
            alpha=alpha
        )

    # --------------------------- 4. Verbose output -------------------------
    if verbose:
        print("\n--- Two-way ANOVA (Type II) - Measure Level ---")
        print(anova_table)

        print("\n--- Tukey HSD: agent_type (pooled across measure_type) ---")
        print(tukey_overall.summary())

        for mtype, tuk in tukey_by_measure.items():
            print(f"\n--- Tukey HSD: agent_type within measure_type = {mtype} ---")
            print(tuk.summary())

    return {
        'long_df'        : long_df,
        'anova'          : anova_table,
        'tukey_overall'  : tukey_overall,
        'tukey_by_measure': tukey_by_measure
    }

if __name__ == '__main__':
    run_two_way_anova_from_person_level()
    run_two_way_anova_from_measure_level()


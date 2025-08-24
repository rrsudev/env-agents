import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm



# ------------------------ Fisher-z utilities ------------------------
def _fisher_z_mean_and_sem(correlation_values):
    """Compute Fisher-z averaged mean and SEM for a list/Series of correlations.

    - clips correlations to (-1 + eps, 1 - eps) to avoid infinities
    - averages in z-space,
    - back-transforms the mean to r
    - propagates the standard error via the derivative of tanh at the mean.
    """
    if correlation_values is None:
        return np.nan, np.nan

    values_array = np.asarray(correlation_values, dtype=float)
    values_array = values_array[~np.isnan(values_array)]

    if values_array.size == 0:
        return np.nan, np.nan

    # Clip to avoid +/-1 which makes arctanh infinite
    epsilon = 1e-8
    clipped_values = np.clip(values_array, -1.0 + epsilon, 1.0 - epsilon)

    z_values = np.arctanh(clipped_values)
    mean_z = np.mean(z_values)
    mean_r = np.tanh(mean_z)

    n = z_values.size
    if n > 1:
        sem_z = np.std(z_values, ddof=1) / np.sqrt(n)
    else:
        sem_z = np.nan

    # Propagate SEM back to r-space via derivative dr/dz = 1 - tanh(z)^2
    if np.isnan(sem_z):
        sem_r = np.nan
    else:
        sem_r = sem_z * (1.0 - mean_r ** 2)

    return mean_r, sem_r

# -----------------------------------------------------------------
# --------------  NORMALIZED CORRELATION PLOTS  -------------------
# -----------------------------------------------------------------
def plot_from_person_level_data():
    # Read the CSV
    df = pd.read_csv('composite_outcomes_person_level_correlations.csv')

    # Define inputs and categories
    inputs = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    label_map = {
        'ALLCONDITIONS': 'Combined',
        'AMERICANVOICES': 'American Voices Project',
        'ENVIRONMENTAL': 'Climate',
        'DEMOGRAPHIC': 'Demographic'
    }
    n_inputs = len(inputs)
    categories = ['env', 'ind']
    category_labels = ['Climate', 'Individual Difference']

    # Set colors for inputs
    colors = {
        'ALLCONDITIONS': 'grey',
        'ENVIRONMENTAL': 'skyblue',
        'AMERICANVOICES': 'salmon',
        'DEMOGRAPHIC': 'lightgreen'
    }

    # Prepare the plot data
    bar_means = []
    bar_errors = []

    for category in categories:
        for input_type in inputs:
            column_name = f'{input_type}_waves_{category}_correlation'
            if column_name in df.columns:
                values = df[column_name].dropna()
                mean, sem = _fisher_z_mean_and_sem(values)
                bar_means.append(mean)
                bar_errors.append(sem)
            else:
                bar_means.append(np.nan)
                bar_errors.append(np.nan)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Number of categories
    n_categories = len(categories)

    # Positions of groups on x-axis
    index = np.arange(n_categories)

    # Width of each bar
    bar_width = 0.2

    # Offsets for each input within a category
    offsets = {
        'ALLCONDITIONS': -2*bar_width,
        'ENVIRONMENTAL': -bar_width,
        'AMERICANVOICES': 0,
        'DEMOGRAPHIC': bar_width
    }

    # Plot each set of bars
    for i, input_type in enumerate(inputs):
        means = bar_means[i::n_inputs]
        errors = bar_errors[i::n_inputs]
        ax.bar(index + offsets[input_type], means, bar_width,
            label=label_map.get(input_type, input_type),
            color=colors[input_type], yerr=errors, capsize=5)

    # Set labels and title
    ax.set_xlabel('Measure Categories')
    ax.set_ylabel('Normalized Correlation')
    ax.set_title('Individual-Level Normalized Correlation by Input Type and Category')
    ax.set_xticks(index)
    ax.set_xticklabels(category_labels)
    ax.legend(title='Agent Type', loc='lower left')

    plt.tight_layout()
    plt.show()


def plot_from_measures_level():

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

    # Read the CSV
    df = pd.read_csv('composite_outcomes_correlations.csv', index_col=0)

    # Inputs and colors
    inputs = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    label_map = {
        'ALLCONDITIONS': 'Combined',
        'AMERICANVOICES': 'American Voices Project',
        'ENVIRONMENTAL': 'Climate',
        'DEMOGRAPHIC': 'Demographic'
    }
    input_colors = {
        'ALLCONDITIONS': 'grey',
        'ENVIRONMENTAL': 'skyblue',
        'AMERICANVOICES': 'salmon',
        'DEMOGRAPHIC': 'lightgreen'
    }

    # Categories mapping
    #categories = {'env': 'Environmental', 'soc': 'Social', 'ind': 'Individual'}
    categories = {'env': 'Climate', 'ind': 'Individual Difference'}

    # Initialize storage for each (category, input)
    data = {cat: {input_: [] for input_ in inputs} for cat in categories.keys()}

    # Process the data
    for index in df.index:
        for filename, info in measures.items():
            for col_pair in info['cols']:
                pred_col = col_pair['pred']
                true_col = col_pair['true']
                expected_string = f"{filename.split('.csv')[0]}_{pred_col}_vs_{true_col}"
                if expected_string == index:
                    for input_ in inputs:
                        colname = f"{input_}_waves_correlation"
                        value = df.loc[index, colname]
                        if not pd.isna(value):
                            data[info['cat']][input_].append(value)

    # Now calculate means and standard errors
    means = {cat: [] for cat in categories.keys()}
    errors = {cat: [] for cat in categories.keys()}

    for cat in categories.keys():
        for input_ in inputs:
            values = data[cat][input_]
            if values:
                mean, error = _fisher_z_mean_and_sem(values)
            else:
                mean = np.nan
                error = np.nan
            means[cat].append(mean)
            errors[cat].append(error)

    print(len(data['ind']['AMERICANVOICES']))
    # Plotting
    x = np.arange(len(categories))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, input_ in enumerate(inputs):
        
        bar = ax.bar(x + idx * width - width,
                [means[cat][idx] for cat in categories.keys()],
                width,
                yerr=[errors[cat][idx] for cat in categories.keys()],
                label=label_map.get(input_, input_),
                color=input_colors[input_],
                capsize=5)


    # Labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized Correlation')
    ax.set_xlabel('Measure Categories')
    ax.set_title('Measure-Level Normalized Correlation by Input Type and Category')
    ax.set_xticks(x)
    ax.set_xticklabels([categories[cat] for cat in categories.keys()])
    ax.legend(title='Agent Type', loc='lower left')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# --------------  RAW CORRELATION PLOTS         -------------------
# -----------------------------------------------------------------

def plot_raw_person():
    df = pd.read_csv('composite_outcomes_person_raw_correlations.csv')

    inputs = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    label_map = {
        'ALLCONDITIONS': 'Combined',
        'AMERICANVOICES': 'American Voices Project',
        'ENVIRONMENTAL': 'Climate',
        'DEMOGRAPHIC': 'Demographic'
    }
    colors = {
        'ALLCONDITIONS': 'grey',
        'ENVIRONMENTAL': 'skyblue',
        'AMERICANVOICES': 'salmon',
        'DEMOGRAPHIC': 'lightgreen'
    }

    categories = ['env', 'ind']
    category_labels = ['Climate', 'Individual Difference']

    all_labels = []
    pred_means = []
    pred_sems = []
    w2_means = []
    w2_sems = []
    bar_colors = []
    hatches = []
    cat_separators = []

    for cat in categories:
        for input_ in inputs:
            filtered = df[(df['agent'] == input_ + '_waves') & (df['category'] == cat)]
            pred_vals = filtered['r_pred_clamped'].dropna()
            w2_vals = filtered['r_w2_clamped'].dropna()

            pred_mean, pred_sem = _fisher_z_mean_and_sem(pred_vals)
            w2_mean, w2_sem = _fisher_z_mean_and_sem(w2_vals)

            pred_means.append(pred_mean)
            pred_sems.append(pred_sem)
            w2_means.append(w2_mean)
            w2_sems.append(w2_sem)

            all_labels.append(f"{label_map[input_]} (Pred)")
            bar_colors.append(colors[input_])
            hatches.append("")
            cat_separators.append(cat)

            all_labels.append(f"{label_map[input_]} (W2)")
            bar_colors.append(colors[input_])
            hatches.append("//")
            cat_separators.append(cat)

    x = np.arange(len(all_labels))
    fig, ax = plt.subplots(figsize=(16, 6))

    for i in range(len(x)):
        mean_val = pred_means[i // 2] if 'Pred' in all_labels[i] else w2_means[i // 2]
        sem_val = pred_sems[i // 2] if 'Pred' in all_labels[i] else w2_sems[i // 2]

        ax.bar(x[i], mean_val, yerr=sem_val, width=0.8,
               color=bar_colors[i], hatch=hatches[i], label=all_labels[i] if i < 8 else "", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Raw Person-Level Correlations (Predicted vs W2)')
    ax.legend(loc='lower left', fontsize='small', title='Agent Type', ncol=2)

    # Vertical separator between categories
    mid_point = len(x) // 2
    ax.axvline(x[mid_point] - 0.5, color='black', linestyle='--', linewidth=1)

    ax.text(mid_point / 2, ax.get_ylim()[1] * 0.95, "Climate", ha='center', fontsize=12, weight='bold')
    ax.text((mid_point + len(x)) / 2, ax.get_ylim()[1] * 0.95, "Individual Difference", ha='center', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()



def plot_raw_measures():
    df = pd.read_csv('composite_outcomes_measure_raw_correlations.csv')

    measures_config = {
        "adaptation_mitigation_output.csv": "env", "cns_output.csv": "env", 
        "envefficacy_match.csv": "env", "envactions_output.csv": "env", "gses_output.csv": "ind",
        "gsjs_output.csv": "ind", "iri_output.csv": "ind", "mfq_output.csv": "ind", "nep_output.csv": "env",
        "nfc_output.csv": "ind", "proximity_output.csv": "env", "risk_aversion_output.csv": "ind",
        "trust_output.csv": "env", "sdo_output.csv": "ind", "mes_composite_output.csv": "ind",
        "emotions_output.csv": "env"
    }

    inputs = ['ALLCONDITIONS', 'ENVIRONMENTAL', 'AMERICANVOICES', 'DEMOGRAPHIC']
    label_map = {
        'ALLCONDITIONS': 'Combined',
        'AMERICANVOICES': 'American Voices Project',
        'ENVIRONMENTAL': 'Climate',
        'DEMOGRAPHIC': 'Demographic'
    }
    colors = {
        'ALLCONDITIONS': 'grey',
        'ENVIRONMENTAL': 'skyblue',
        'AMERICANVOICES': 'salmon',
        'DEMOGRAPHIC': 'lightgreen'
    }

    categories = {'env': 'Climate', 'ind': 'Individual Difference'}
    cat_keys = list(categories.keys())

    all_labels = []
    pred_means = []
    pred_sems = []
    w2_means = []
    w2_sems = []
    bar_colors = []
    hatches = []
    cat_separators = []

    for c_idx, cat in enumerate(cat_keys):
        for input_ in inputs:
            filtered = df[(df['agent'] == input_ + '_waves') &
                          (df['measure_file'].map(measures_config.get) == cat)]

            pred_vals = filtered['r_pred_clamped'].dropna()
            w2_vals = filtered['r_w2_clamped'].dropna()

            pred_mean, pred_sem = _fisher_z_mean_and_sem(pred_vals)
            w2_mean, w2_sem = _fisher_z_mean_and_sem(w2_vals)

            pred_means.append(pred_mean)
            pred_sems.append(pred_sem)
            w2_means.append(w2_mean)
            w2_sems.append(w2_sem)

            all_labels.append(f"{label_map[input_]} (Pred)")
            bar_colors.append(colors[input_])
            hatches.append("")
            cat_separators.append(categories[cat])

            all_labels.append(f"{label_map[input_]} (W2)")
            bar_colors.append(colors[input_])
            hatches.append("//")
            cat_separators.append(categories[cat])

    x = np.arange(len(all_labels))  # bar positions
    fig, ax = plt.subplots(figsize=(16, 6))

    for i in range(len(x)):
        mean_val = pred_means[i // 2] if 'Pred' in all_labels[i] else w2_means[i // 2]
        sem_val = pred_sems[i // 2] if 'Pred' in all_labels[i] else w2_sems[i // 2]

        ax.bar(x[i], mean_val, yerr=sem_val, width=0.8,
               color=bar_colors[i], hatch=hatches[i], label=all_labels[i] if i < 8 else "", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_ylabel('Average Correlation')
    ax.set_title('Raw Measure-Level Correlations (Predicted vs W2)')
    ax.legend(loc='lower left', fontsize='small', title='Agent Type', ncol=2)

    # Vertical separator between categories
    mid_point = len(x) // 2
    ax.axvline(x[mid_point] - 0.5, color='black', linestyle='--', linewidth=1)

    ax.text(mid_point / 2, ax.get_ylim()[1] * 0.95, "Climate", ha='center', fontsize=12, weight='bold')
    ax.text((mid_point + len(x)) / 2, ax.get_ylim()[1] * 0.95, "Individual Difference", ha='center', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_from_measures_level()
    plot_from_person_level_data()
    plot_raw_person()
    plot_raw_measures()

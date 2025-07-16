import pandas as pd
import numpy as np

# CONFIGURE FILE PATHS HERE
INPUT_CSV = "apr8analysispipeline/may5wave1_ground_truth.csv"     
OUTPUT_CSV = "may5wave1_numground_truth.csv" 

# DEFINE MAPPINGS
likert_map_7pt = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3, "Unsure": 4,
    "Somewhat agree": 5, "Agree": 6, "Strongly agree": 7,
    "Neutral": 4,

}

likert_map_5pt = {
    "Strongly disagree": 1, "Disagree": 2, "Unsure": 3,
    "Agree": 4, "Strongly agree": 5,
    "Strongly disagree": 1,
    "Somewhat agree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat disagree": 4,
    "Strongly agree": 5

}

ecdc_map = {
    "Strongly disagree": 1, "Somewhat agree": 2,
    "Neither agree nor disagree": 3, "Somewhat disagree": 4,
    "Strongly agree": 5
}

env_actions_map = {
    "Very unlikely": 1,
    "Unlikely": 2,
    "Somewhat unlikely": 3,
    "Neutral": 4,
    "Somewhat likely": 5,
    "Likely": 6,
    "Very likely": 7,
    "I already do this": 7
}

mfq_map = {
    '(0) Not at all relevant (This consideration has nothing to do with my judgments of right and wrong.)': 0,
    '(1) Not very relevant': 1, '(2) Slightly relevant': 2, '(3) Somewhat relevant': 3,
    '(4) Very relevant': 4, '(5) Extremely relevant (This is one of the most important factors when I judge right and wrong.)': 5,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
}

gse_likert_map = {
    "(1) Not at all true": 1, "(2) Hardly true": 2,
    "(3) Moderately true": 3, "(4) Exactly true": 4
}

nfc_likert_map = {
    "(1) Completely disagree": 1, "-2": 2, "-3": 3,
    "-4": 4, "-5": 5, "(6) Completely agree": 6
}

svi_scale_map = {
    "Opposed to my principles (0)": 0, "Not important (1)": 1, "Important (4)": 4, "Of supreme importance (8)": 8,
    **{str(i): i for i in range(9)}, **{f"-{i}": i for i in range(1, 9)}
}

ss_map = {
    'None at all': 1, 'A little': 2, 'A moderate amount': 3, 'A lot': 4, 'A great deal': 5
}

emotions_map = {
    'Yes': 1, 'No': 2, 'Prefer Not to Say': 3
}

efficacy_map = {
    'Strongly disagree': 1, 'Somewhat disagree': 2,
    'Neither agree nor disagree': 3, 'Somewhat agree': 4,
    'Strongly agree': 5
}

sdo_scale_map = {
    "(1) Strongly Oppose": 1, "(2) Somewhat Oppose": 2,
    "(3) Slightly Oppose": 3, "(4) Neutral": 4,
    "(5) Slightly Favor": 5, "(6) Somewhat Favor": 6,
    "(7) Strongly Favor": 7
}

gsjs_scale_map = {
    "Strongly disagree": 1, "Disagree": 2, "Moderately disagree": 3,
    "Mildly disagree": 4, "Neither agree nor disagree": 5,
    "Mildly agree": 6, "Moderately agree": 7,
    "Agree": 8, "Strongly Agree": 9
}

direct_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
reverse_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
iri_mapping = {
    "IRI_1":  ("FS", False), "IRI_2":  ("EC", False), "IRI_3":  ("PT", True),  "IRI_4":  ("EC", True),
    "IRI_5":  ("FS", False), "IRI_6":  ("PD", False), "IRI_7":  ("FS", True),  "IRI_8":  ("PT", False),
    "IRI_9":  ("EC", False), "IRI_10": ("PD", False), "IRI_11": ("PT", False), "IRI_12": ("FS", True),
    "IRI_13": ("PD", True),  "IRI_14": ("EC", True),  "IRI_15": ("PT", True),  "IRI_16": ("FS", False),
    "IRI_17": ("PD", False), "IRI_18": ("EC", True),  "IRI_19": ("PD", True),  "IRI_20": ("EC", False),
    "IRI_21": ("PT", False), "IRI_22": ("EC", False), "IRI_23": ("FS", False), "IRI_24": ("PD", False),
    "IRI_25": ("PT", False), "IRI_26": ("FS", False), "IRI_27": ("PD", False), "IRI_28": ("PT", False),
}

def map_choice(val):
    if isinstance(val, str):
        val = val.lower()
        if "option a" in val:
            return "A"
        elif "option b" in val:
            return "B"
    return np.nan

def map_column(col, series):
    if "PSYCHPROX" in col:
        return series
    elif "NEPS" in col:
        return series.map(likert_map_5pt)
    elif "ECDC" in col:
        return series.map(ecdc_map)
    elif "ENV_ACTIONS" in col:
        return series.map(env_actions_map)
    elif "CONNECTNATURE" in col:
        return series.map(likert_map_7pt)
    elif "MFQ" in col:
        return series.map(mfq_map)
    elif "GSES" in col:
        return series.map(gse_likert_map)
    elif "IRI" in col:
        is_reversed = iri_mapping.get(col, (None, False))[1]
        return series.map(reverse_map if is_reversed else direct_map)
    elif "NFC" in col:
        return series.map(nfc_likert_map)
    elif "SVI" in col:
        return series.map(svi_scale_map)
    elif "SS" in col:
        return series.map(ss_map)
    elif "emotions" in col.lower():
        return series.map(emotions_map)
    elif "RA" in col:
        return series.apply(map_choice)
    elif "EFFICACY" in col:
        return series.map(efficacy_map)
    elif "SDO" in col:
        return series.map(sdo_scale_map)
    elif "GSJS" in col:
        return series.map(gsjs_scale_map)
    elif "Adaptation" in col or "Mitigation" in col:
        return series.map(likert_map_7pt)
    else:
        return series  

# MAIN
def main():
    df = pd.read_csv(INPUT_CSV)
    df_mapped = df.copy()
    for col in df.columns:
        df_mapped[col] = map_column(col, df[col])
    df_mapped.to_csv(OUTPUT_CSV, index=False)
    print(f"Mapped CSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

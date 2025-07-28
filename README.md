# Agent Simulations of Individuals' Climate Attitudes 🌎

Code release for the climate attitudes generative agents analysis pipeline. 

## STEP 0 PREPROCESSING
Note that the preprocessing steps are released for transparency. You do not need to run these scripts with the data format you are provided with. 

#### 0.1: Agent Construction and Ground Truth Collection
Before we run the code presented here, we collect participant interview responses from the GABM platform and survey responses from Qualtrics. We export the raw survey data to act as the ground truth. 

Agents are constructed from interview transcripts and all responses are stored in JSON files.


#### 0.2: Cleaning Raw Data
We clean the ground truth information and separate our simulated data into two waves before processing.

`make_test_retest_csv`: clean the ground truth data exported from the Qualtrics survey for use in later analysis.

`make_simulated_csv`: combine and restructure JSON files of simulated agent outputs into a csv file for comparison with ground truth data. 

#### 0.3: Data Release
`0_data_release` contains `data_release.zip`, where you can find anonymized pilot data for both human ground truth and simulated responses.

## STEP 1: Measure-Level Analysis
`all_measures.py` : Analyze data for each of the climate and individual difference measures used. Output csv files showing Wave 1 (W1), Wave 2 (W2) and predicted values for each measure, including subscales where applicable. Calculate composites for scales, per respondent. 

To proceed with our pipeline, ensure you run all measures across each of the four conditions. You would need to run something to the effect of:

```
  all_measures.py all --agent ALLCONDITIONS
  all_measures.py all --agent AMERICANVOICES
  all_measures.py all --agent DEMOGRAPHIC
  all_measures.py all --agent ENVIRONMENTAL
```

Below is a summary of all measures covered.

| Category             | Measure (name in paper)                                | Pipeline function         | Output CSV                        |
|----------------------|--------------------------------------------------------|----------------------------|-----------------------------------|
| **Climate**          | New Ecological Paradigm Scale                          | `run_nep`                  | `nep_output.csv`                  |
|                      | Adaptation and Mitigation Behavioral Intentions        | `run_adaptation_mitigation`| `adaptation_mitigation_output.csv`|
|                      | Connectedness to Nature Scale                          | `run_cns`                  | `cns_output.csv`                  |
|                      | Environmental Efficacy                                 | `run_envefficacy`          | `envefficacy_match.csv`           |
|                      | Relative Carbon Mitigation Impact                      | *(pipeline TBD)*           | *(csv TBD)*                       |
|                      | Psychological Proximity of Climate Change              | `run_proximity`            | `proximity_output.csv`            |
|                      | Climate Behavioral Adoption Likelihood                 | `run_envactions`           | `envactions_output.csv `          |
|                      | Climate Emotions                                       | `run_emotions`             | `emotions_output.csv`             |
|                      | Trust in Science and Climate Scientists                | `run_trust`                | `trust_output.csv`                |
| **Individual Difference** | Risk Aversion                                     | `run_risk_aversion`        | `risk_aversion_output.csv`        |
|                      | General Self-Efficacy Scale                            | `run_gses`                 | `gses_output.csv`                 |
|                      | Moral Foundations Questionnaire                        | `run_mfq`                  | `mfq_output.csv`                  |
|                      | Moral Expansiveness Scale                              | `run_mes`                  | `mes_composite_output.csv`        |
|                      | Need For Closure Scale                                 | `run_nfc`                  | `nfc_output.csv`                  |
|                      | Social Dominance Orientation                           | `run_sdo`                  | `sdo_output.csv`                  |
|                      | Interpersonal Reactivity Index                         | `run_iri`                  | `iri_output.csv`                  |
|                      | General System Justification Scale                     | `run_gsjs`                 | `gsjs_output.csv`                 |



## STEP 2: Calculate and Plot Normalized Correlations
`composite_analysis.py` takes the output of `all_measures.py` and conducts measure level and individual level analysis. 
`plot_composite_analysis.py` can be used to visualize the output of `composite_analysis.py`.
### Measure Level
- Takes the correlation for each composite measure given each input type
- Plots the average correlation for composites in each category (along with confidence interval)

### Individual Level
- Takes the data points for one participant in a given category and calculate the correlation for that specific participant in that specific category
- Plots the average participant-level correlation coefficient for each question category and input type

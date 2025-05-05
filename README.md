# env-agents

Environmental generative agents analysis pipeline. 

## Agent Construction and Ground Truth Collection
Before the code presented here, we collect participant interview responses from the GABM platform and survey responses from Qualtrics. We export the survey data to act as the ground truth. 

Agents are constructed from interview transcripts and all responses are stored in JSON files.


## Cleaning Raw Data
We clean the ground truth information before processing.

`clean_ground_truth`: clean the ground truth data exported from the Qualtrics survey for use in later analysis.

`make_simulated_csv`: combine and restructure JSON files of simulated agent outputs into a csv file for comparison with ground truth data. 


## Measure-Level Analysis
Analyze data separated by each of the 17 measures used. Output csv files comparing truth and predicted values for each measure. Calculate composites for scales, and will calculate normalized accuracy for non-scales. [Measures by name](https://docs.google.com/spreadsheets/d/1Md7Z_8UZ0GH0gXQ6aLsMQwhetOjc1jj-ZBrHuE8KOcc/edit?gid=1648875056#gid=1648875056)


## Overall Analysis
Composite Metrics
- Take the correlation for each composite measure given each input type
- Plot the average correlation for composites in each category (along with confidence interval)


Respondent Level
- Take the data points for one participant in a given category and calculate the correlation for that specific participant in that specific category
- Plot the average participant-level correlation coefficient for each question category and input type


## Future Steps
- Possible refactoring of measure level analysis
- Normalized accuracy for non-scales
- Calculate cronbach’s alpha

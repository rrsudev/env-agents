import sys
import os
from get_survey_questions import *
from writers import *
import multiprocessing

# Get the absolute path of the analysis_code directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "analysis_code")))
from source.data_preprocessing.unzip_interview import *
from source.experiments.categorical.categorical_generations import *


# Preparation
def preparation(platform):
    # unzip interview transcripts
    unzip_interview_transcript(platform)


def process_chunk(chunk):
    counter, process, agent_id, question, platform = chunk
    agent_str = read_file_to_string(
        f"../agent_bank/source_data/{platform}/gabm_infra/interview_transcript/data/{agent_id}/interview.txt")  # the interview
    question_pt = ""
    question_pt += f"Q: {question['Question']}\n"
    question_pt += f"Option: {question['Options']}\n\n"
    question_id = question['Question ID']

    r = run_gpt_generate_categorical_response(
        agent_str,
        question_pt,
        prompt_version="5",
        gpt_version="GPT4o",
        test_input=None,
        verbose=False,
        singular=True)

    response = r[0]
    prompt_out = r[1][1]

    directory = f"../results/{platform}/{agent_id}/{question_id}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # response = process_response(response)
    write_prompt(prompt_out, f"{directory}/prompt.txt")
    write_json_to_file(response, f"{directory}/response.json")


def get_demographic_string(platform, agent_id, demographic_list):
    for agent in demographic_list:
        if agent['email'] == agent_id:
            agent_info = ""
            for key in agent.keys():
                if key != "email":
                    agent_info += f"{key}: {agent[key]}\n"
            return agent_info
    return None

def csv_to_list_demographics(file_path, demographics):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Automatically maps columns to values
        data_list = []
        for row in reader:
            filtered_row = {key: row[key] for key in demographics if
                            key in row.keys()}  # Keep only specified columns
            filtered_row["email"] = row["email"]
            data_list.append(filtered_row)
    return data_list


def process_chunk_multi_q(chunk):
    counter, process, agent_id, questions, platform, climate_interview, demographic, AVP = chunk

    agent_str = ""

    if AVP:
        agent_str += "INTERVIEW: \n"
        agent_str += read_file_to_string(
            f"../../LLM Agent Simulation/Agent Data/{platform}/{agent_id}/memory/gabm_infra/interview.txt")  # the interview
        agent_str += "----------------------------------\n\n"
    if climate_interview:
        agent_str += "INTERVIEW: \n"
        agent_str += read_file_to_string(
            f"../agent_bank/source_data/{platform}/gabm_infra/interview_transcript/data/{agent_id}/interview.txt")  # the interview
        agent_str += "----------------------------------\n\n"
    if demographic:
        demographic_list = csv_to_list_demographics("new_analysis_summaries/demographic_summary.csv", ["political_ideology", "race", "gender", "age"])
        agent_str += "PARTICIPANT DEMOGRAPHIC INFORMATION: \n"
        agent_str += get_demographic_string(platform, agent_id, demographic_list)
        if agent_str is None:
            print(f"NO AGENT NAMED {agent_id} FOR DEMOGRAPHICS")
            return

    question_pt = ""
    question_id = questions[0]['Question ID']
    for question in questions:
        question_pt += f"Q: {question['Question']}\n"
        question_pt += f"Option: {question['Options']}\n\n"

    r = run_gpt_generate_categorical_response(
        agent_str,
        question_pt,
        prompt_version="5",
        gpt_version="GPT4o",
        test_input=None,
        verbose=False,
        singular=False)

    response = r[0]
    prompt_out = r[1][1]

    partial_directory = ""

    if AVP:
        partial_directory +="baseInterview_"
    if climate_interview:
        partial_directory += "climateInterview_"
    if demographic:
        partial_directory += "demographic"
    directory = f"../results/{partial_directory}/{platform}/{agent_id}/{question_id}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # response = process_response(response)
    write_prompt(prompt_out, f"{directory}/prompt.txt")
    write_json_to_file(response, f"{directory}/response.json")


def run_simulation(platform, climate_interview, demographic, AVP):
    parent_path = f"../agent_bank/source_data/{platform}/gabm_infra/interview_transcript/data"
    questions = csv_to_list_of_dicts("../master_questions.csv")
    agents = [entry for entry in os.listdir(parent_path)
              if os.path.isdir(os.path.join(parent_path, entry)) and not entry.startswith('.')]

    chunk_info_list = []
    num_processes = 50
    counter = 0

    for agent in agents:
        q_list = []
        for i in range(len(questions)):
            q_list.append(questions[i])
            if (i+1) % 5 == 0:
                chunk_info_list.append((counter, counter % num_processes, agent, q_list, platform, climate_interview, demographic, AVP))
                counter += 1
                q_list = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_chunk_multi_q, chunk_info_list)

def test():
    questions = csv_to_list_of_dicts("../master_questions.csv")
    agent = "test@email.com"
    q_set_1 = []
    q_set_2 = []
    for i in range(len(questions)):
        if i < 10 and i >=5:
            q_set_1.append(questions[i])
        if i < 15 and i >= 10:
            q_set_2.append(questions[i])

    chunk = 0, 0, agent, q_set_1, "prolific"
    process_chunk_multi_q(chunk)
    chunk = 0, 0, agent, q_set_2, "prolific"
    process_chunk_multi_q(chunk)




if __name__ == '__main__':
    #test()

    # Uncomment these lines
    # (1) zipped agent files should be in agent_bank/source_data/{Bovitz or Prolific}/gabm_infra/interview_transcript/zipped
    # (2) Set the platform here
    os.chdir('./analysis_code')
    platform = "bovitz"
    climate_interview = False
    demographic = True
    AVP = True
    #preparation(platform)
    run_simulation(platform, climate_interview, demographic, AVP)

    # code to count agents and number of simulated questions run
    # parent_path = "results/demographic/bovitz"
    # agents = [entry for entry in os.listdir(parent_path)
    #           if os.path.isdir(os.path.join(parent_path, entry)) and not entry.startswith('.')]
    # print(len(agents), " agents")
    # for agent in agents:
    #     cur_path = f"{parent_path}/{agent}"
    #     dirs = [entry for entry in os.listdir(cur_path)
    #           if os.path.isdir(os.path.join(cur_path, entry)) and not entry.startswith('.')]
    #     print("Agent: ", agent, " Dirs: ", len(dirs))


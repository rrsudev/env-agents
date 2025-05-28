import os
from source.global_methods import *

base_path = os.path.dirname(os.path.dirname(__file__))
analysis_decimal_round = 2

def get_fin_participant_email_list(rand_count=None, rand_order_v=1): 
  if not rand_count: 
    # p_list = read_file_to_list(f"{base_path}/agent_bank/source_data/bovitz/gabm_infra/participant_list.csv", 
    #            header=False, strip_trail=True)

    # all_valid_emails = [row[2].lower().strip() for row in p_list]
    # return all_valid_emails
    '''
    p_list = read_file_to_list(f"{base_path}/agent_bank/source_data/bovitz/gabm_infra/participant_list_rand_order_v{rand_order_v}.csv", 
               header=False, strip_trail=True)
    '''
    p_list = read_file_to_list(f"{base_path}/agent_bank/source_data/prolific/participant_list.csv", header=False,
                               strip_trail=True)
    p_list = [[row[0], int(row[1])] for row in p_list]

    p_list = sorted(p_list, key=lambda x: x[1])
    all_valid_emails = [row[0].lower().strip() for row in p_list]
    return all_valid_emails
    
  else:
    '''
    p_list = read_file_to_list(f"{base_path}/agent_bank/source_data/bovitz/gabm_infra/participant_list_rand_order_v{rand_order_v}.csv", 
               header=False, strip_trail=True)
    '''

    p_list = read_file_to_list(f"{base_path}/agent_bank/source_data/prolific/participant_list.csv", header=False,
                               strip_trail=True)
    p_list = [[row[0], int(row[1])] for row in p_list]

    p_list = sorted(p_list, key=lambda x: x[1])[:rand_count]
    all_valid_emails = [row[0].lower().strip() for row in p_list]
    return all_valid_emails


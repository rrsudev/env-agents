import pandas as pd
import os
import zipfile
import openai
import json

import multiprocessing
from multiprocessing import Pool

from datetime import datetime
from source.global_methods import *
from source.utils import *

from source.llm_utils.gpt_structure import *
from source.llm_utils.print_prompt import *


def run_gpt_generate_categorical_response(
    agent_str, 
    question_pt, 
    prompt_version="5", 
    gpt_version="GPT4o",  
    test_input=None, 
    verbose=False,
    singular=False):
  def create_prompt_input(agent_str, question_pt, test_input=None):
    prompt_input = [agent_str, question_pt]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      return True 
    except:
      return False 

  def get_fail_safe():
    return None

  if not singular: 
    prompt_template = f"source/llm_utils/prompt_template/categorical_response/respond_batch_q_v{prompt_version}.txt" 
  else: 
    prompt_template = f"source/llm_utils/prompt_template/categorical_response/respond_batch_q_v{prompt_version}_singular.txt" 

  prompt_input = create_prompt_input(agent_str, question_pt) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  # output = chat_safe_generate(prompt, gpt_version, 2, fail_safe,
  #                       __chat_func_validate, __chat_func_clean_up, verbose)
  output = generate_response(prompt)

  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]


def generate_response(prompt):
  response = "Error"
  while response == "Error":
    #openai.api_key = '
    openai.api_key = 'enter key here'
    completion = openai.chat.completions.create(
      model="gpt-4o-mini",  # Make sure you use the correct model ID for GPT-4o
      messages=[{"role": "user", "content": prompt}]
    )
    response = process_response(completion.choices[0].message.content)
  return response


def process_response(response_text):
  if response_text.startswith("```json"):
    response_text = response_text[7:-3].strip()  # Remove the ```json and ```

  # Parse the JSON string
  try:
    json_response = json.loads(response_text)
    # print(json.dumps(json_response, indent=4))  # Pretty print the JSON, if needed
    return json_response
  except json.JSONDecodeError as e:
    print("Failed to decode JSON:", e)
    print("Raw response text:", response_text)
    return "Error"





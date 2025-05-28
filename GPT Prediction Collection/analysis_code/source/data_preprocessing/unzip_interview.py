import pandas as pd
import os
import zipfile

from datetime import datetime
from source.global_methods import *
from source.utils import *


def unzip_interview_transcript(source_data="bovitz"): 
  print ("<Unzipping interview transcript from GABM Infra.>")

  # Define the path to the zip directory
  zip_dir = f"{base_path}/../agent_bank/source_data/{source_data}/gabm_infra/interview_transcript/zipped"
  unzipped_dir = f"{base_path}/../agent_bank/source_data/{source_data}/gabm_infra/interview_transcript/data"

  # Create the unzipped directory if it doesn't exist
  if not os.path.exists(unzipped_dir):
    os.makedirs(unzipped_dir)

  # Iterate over each file in the zip directory
  for count, filename in enumerate(os.listdir(zip_dir)):
    if filename != ".DS_Store": 
      new_filename = ".".join(filename.split(".")[:-1]).lower()
      curr_unzipped_dir = os.path.join(unzipped_dir, new_filename)
      if os.path.exists(curr_unzipped_dir):
        print (f"File {count}: Unzipped folder already exists at: {curr_unzipped_dir}")

      else: 
        if filename.endswith('.zip'):
          print (f"File {count}: Unzipping folder at: {new_filename}")
          zip_path = os.path.join(zip_dir, filename)
          with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(curr_unzipped_dir)

  print ()
  print("All files have been unzipped to the 'unzipped' directory.")


if __name__ == '__main__':
  update_agent_storage("prolific_pilot_interview")















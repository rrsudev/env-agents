import json

def write_prompt(data, file_path):
    with open(file_path, 'w') as file:
        file.write(data)

# Function to write JSON to a file
def write_json_to_file(json_data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)  # Pretty print the JSON
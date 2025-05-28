import csv

def csv_to_list_of_dicts(file_path):
    """
    Reads a CSV file and converts it into a list of dictionaries.
    Each row in the CSV becomes a dictionary, where keys are column names.
    """
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Automatically maps columns to values
        data_list = [row for row in reader]  # Convert reader object to list of dicts
    return data_list

# Example usage
if __name__ == "__main__":
    file_path = "example.csv"  # Change this to your CSV file path
    data = csv_to_list_of_dicts(file_path)
    print(data)
import json

# Define the path to your JSON file
file_path = 'instruct_tuning/instruct.json'

# Open and load the JSON file
try:
    with open(file_path, 'r') as file:
        data = json.load(file)  # Parse JSON file into a Python dictionary
        print("File content loaded successfully!")
        breakpoint()
        print(data)  # Print the content of the JSON file
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")
except json.JSONDecodeError:
    print("Error decoding JSON. Make sure the file has valid JSON content.")
except Exception as e:
    print(f"An error occurred: {e}")
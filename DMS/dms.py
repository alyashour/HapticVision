# data management system
# uses JSON
import json
import os

def save_data(data, filename: str):
    """
    Saves data to a file in the Appdata folder with the given filename.
    :param data:
    :param filename:
    :return:
    """
    path = f'../Appdata/{filename}'
    with open(path, 'w') as file:
        json.dump(data, file)

def load_data(filename: str):
    path = f'../Appdata/{filename}'
    # check if the file exists
    if os.path.exists(path):
        with open(path) as file:
            return json.load(file)
    else:
        raise FileNotFoundError
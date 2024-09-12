import csv
import ast

import math

avg_hand_length = 0.05 # in meters

def get_magnitude(list):
    sum = 0
    for item in list:
        sum += item * item
    return math.sqrt(sum)

# Open the CSV file
rows = []
with open('../Output/left_hand_velocities.csv', mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Optional: Read the header row (if it exists)
    header = next(csv_reader)

    # Iterate over the rows in the CSV
    for row in csv_reader:
        rows.append(row) # Each row is a list of values

total_distance = 0
for row in rows:
    for cell in row:
        try:
            parsed_list = ast.literal_eval(cell)
            speed = get_magnitude(parsed_list)
            frame_time = 1/30 # in seconds
            distance = speed * frame_time
            total_distance += distance * avg_hand_length
        except SyntaxError:
            pass

print(f'Total Distance: {total_distance: .2f}m')
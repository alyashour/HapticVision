import pandas as pd
import matplotlib.pyplot as plt

config = {
    'do_smoothing': True,
    'smoothing_window_width': 5
}

# Load the data from CSV file
csv_file = 'Output/output.csv'
df = pd.read_csv(csv_file)

# smooth the data if needed
if config['do_smoothing']:
    df = df.rolling(window=config['smoothing_window_width'], center=True).mean()
    # drop NaN values created by rolling at the edges
    df = df.dropna().reset_index(drop=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['frame'], df['thumb_speed'], label='Thumb', color='blue')
plt.plot(df['frame'], df['pointer_speed'], label='Pointer', color='green')
plt.plot(df['frame'], df['middle_speed'], label='Middle', color='red')
plt.plot(df['frame'], df['ring_speed'], label='Ring', color='purple')
plt.plot(df['frame'], df['pinky_speed'], label='Pinky', color='orange')

# Adding labels and title
plt.xlabel('Frame Number')
plt.ylabel('Speed (magnitude)')
plt.title('Fingertip Speeds')
plt.legend()

# Display the plot
plt.show()

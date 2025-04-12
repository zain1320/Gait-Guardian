import pandas as pd
import matplotlib.pyplot as plt

# Define the expected column names
columns = ['timestamp', 'label', 'accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 
           'pitch', 'roll', 'bigToe', 'medialFore', 'lateralFore', 'heel', 'gaitPhase']

# Load the CSV, specifying that it has no header and providing column names
df = pd.read_csv('normal.csv', header=None, names=columns)

# Verify the DataFrame
print("First few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# Plot accelX over timestamp
plt.plot(df['timestamp'], df['accelX'])
plt.xlabel('Timestamp')
plt.ylabel('Acceleration X')
plt.title('Acceleration X over Time')
plt.show()
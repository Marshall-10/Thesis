import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the CSV file
file_path = r'C:\Users\Hussein\OneDrive\Desktop\RB Pi\CSI_Dataset\Data\individual_3\Leo_24-06-2024_12_23_00(txt).txt'

# Read the data, ensure 'amplitude' and 'phase' are read as strings
data = pd.read_csv(file_path, sep=';', dtype={'amplitude': str, 'phase': str})

# Clean and preprocess the data
data['amplitude'] = data['amplitude'].str.replace('.', '', regex=False).astype(float)
data['phase'] = data['phase'].str.replace('.', '', regex=False).astype(float)

# Extract amplitude and phase
amplitude = data['amplitude']
phase = data['phase']

# Plot amplitude and phase
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(amplitude, label='Amplitude')
plt.title('Amplitude')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(phase, label='Phase', color='orange')
plt.title('Phase')
plt.xlabel('Sample Index')
plt.ylabel('Phase')
plt.legend()

plt.tight_layout()
plt.show()

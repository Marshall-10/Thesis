import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the data
file_path = r'C:\Users\Hussein\OneDrive\Desktop\RB Pi\CSI_Dataset\Data\individual_3\Leo_24-06-2024_12_23_00(txt).txt'

# Read the data
data = pd.read_csv(file_path, delimiter=';', usecols=['timestamp', 'amplitude'])

# Function to fix timestamp format
def fix_timestamp_format(ts):
    date, time = ts.split(' ')
    time_parts = time.rsplit(':', 1)
    corrected_time = time_parts[0] + '.' + time_parts[1]
    corrected_timestamp = f"{date} {corrected_time}"
    return corrected_timestamp

# Apply the fix to the timestamp column
data['timestamp'] = data['timestamp'].apply(fix_timestamp_format)

# Convert timestamp to a numeric format
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

# Ensure the data is sorted by timestamp
data = data.sort_values(by='timestamp')

# Function to add noise
def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Function to scale the data
def scale_data(data, scale_factor=1.5):
    return data * scale_factor

# Function to time warp the data
def time_warp(data, sigma=0.1):
    x = np.arange(len(data))
    tt = np.cumsum(np.random.normal(1, sigma, len(data)))
    tt = tt / tt[-1] * (len(data) - 1)
    interp = interp1d(tt, data, kind='linear', fill_value="extrapolate")
    return interp(x)

# Function to slice the signal at regular intervals
def slice_signal_at_intervals(data, interval=15, slice_duration=5):
    sliced_data = []
    total_duration = data['timestamp'].max()
    current_time = 0
    while current_time + slice_duration <= total_duration:
        start_time = current_time
        end_time = current_time + slice_duration
        slice_segment = data[(data['timestamp'] >= start_time) & (data['timestamp'] < end_time)]
        if not slice_segment.empty:
            sliced_data.append(slice_segment)
        current_time += interval
    return sliced_data

# Function to reverse slice the signal, keeping parts not included in slices
def reverse_slice_signal_at_intervals(data, interval=15, slice_duration=5):
    mask = np.ones(len(data), dtype=bool)
    total_duration = data['timestamp'].max()
    current_time = 0
    while current_time + slice_duration <= total_duration:
        start_time = current_time
        end_time = current_time + slice_duration
        mask[(data['timestamp'] >= start_time) & (data['timestamp'] < end_time)] = False
        current_time += interval
    reverse_sliced_data = data[mask]
    return reverse_sliced_data

# Extract time and amplitude data
time = data['timestamp'].values
amplitude = data['amplitude'].values

# Apply augmentations
data_noisy = add_noise(amplitude)
data_scaled = scale_data(amplitude)
data_time_warped = time_warp(amplitude)
sliced_data = slice_signal_at_intervals(data)
reverse_sliced_data = reverse_slice_signal_at_intervals(data)

# Plotting
plt.figure(figsize=(16, 8))

plt.subplot(231)
plt.plot(time, amplitude, color='purple')
plt.title('Original')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(232)
plt.plot(time, data_noisy, color='purple')
plt.title('Noisy')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(233)
plt.plot(time, data_scaled, color='purple')
plt.title('Scaled')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(234)
plt.plot(time, data_time_warped, color='purple')
plt.title('Time Warped')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(235)
for segment in sliced_data:
    plt.plot(segment['timestamp'], segment['amplitude'], color='blue')
plt.title('Sliced')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(236)
plt.plot(reverse_sliced_data['timestamp'], reverse_sliced_data['amplitude'], color='red')
plt.title('Reverse Sliced')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

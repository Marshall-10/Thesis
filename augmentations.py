import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Amplitude augmentations
def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, scale_factor=1.5):
    return data * scale_factor

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

# Phase augmentations
def add_horizontal_jitter(time, jitter_level=0.25):
    jitter = np.random.uniform(-jitter_level, jitter_level, time.shape)
    return time + jitter

def invert_data(data):
    return -data

def positive_data(data):
    return np.maximum(0, data)

def negative_data(data):
    return np.minimum(0, data)

# Synthetic data generation function
def generate_synthetic_data(amplitude, phase, num_samples=1000):
    synthetic_amplitude = np.random.choice(amplitude, num_samples)
    synthetic_phase = np.random.choice(phase, num_samples)
    return synthetic_amplitude, synthetic_phase

# Function to fix timestamp format
def fix_timestamp_format(ts):
    try:
        date, time = ts.split(' ')
        time_parts = time.rsplit(':', 1)
        corrected_time = time_parts[0].replace('.', ':') + '.' + time_parts[1]
        corrected_timestamp = f"{date} {corrected_time}"
        return corrected_timestamp
    except IndexError as e:
        print(f"Error processing timestamp: {ts} - {e}")
        return ts

# Function to apply all augmentations and generate data
def apply_augmentations(file_path):
    data = pd.read_csv(file_path, delimiter=';', usecols=['timestamp', 'amplitude', 'phase'])
    data['timestamp'] = data['timestamp'].apply(fix_timestamp_format)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()
    data = data.sort_values(by='timestamp')

    augmentations = {}
    augmentations['noisy'] = data.copy()
    augmentations['noisy']['amplitude'] = add_noise(data['amplitude'].values)
    augmentations['scaled'] = data.copy()
    augmentations['scaled']['amplitude'] = scale_data(data['amplitude'].values)
    augmentations['sliced'] = slice_signal_at_intervals(data)
    augmentations['reverse_sliced'] = reverse_slice_signal_at_intervals(data)
    augmentations['jitter'] = data.copy()
    augmentations['jitter']['timestamp'] = add_horizontal_jitter(data['timestamp'].values)
    augmentations['inverted'] = data.copy()
    augmentations['inverted']['phase'] = invert_data(data['phase'].values)
    augmentations['positive'] = data.copy()
    augmentations['positive']['phase'] = positive_data(data['phase'].values)
    augmentations['negative'] = data.copy()
    augmentations['negative']['phase'] = negative_data(data['phase'].values)

    synthetic_amplitude, synthetic_phase = generate_synthetic_data(data['amplitude'].values, data['phase'].values)
    augmentations['synthetic'] = pd.DataFrame({'timestamp': np.arange(len(synthetic_amplitude)), 'amplitude': synthetic_amplitude, 'phase': synthetic_phase})

    return augmentations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import glob
from tqdm import tqdm
from model.res_net_use_this import ResNetWithDropoutAndBatchNorm, ResidualBlock

# Hyperparameters
batch_size = 32

# Function to handle nan or inf values in the data
def handle_nan_inf(df):
    if df.isnull().values.any() or np.isinf(df.values).any():
        print("Data contains nan or inf values. Handling them...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
    return df

# Function to normalize the data
def normalize_data(df):
    df = handle_nan_inf(df)
    return (df - df.mean()) / df.std()

# Function to load and preprocess data from multiple files
def load_and_preprocess_data(files):
    data_list = []

    for file in files:
        try:
            data = pd.read_csv(file, delimiter=';')
            print(f"Loaded file: {file}")
        except PermissionError as e:
            print(f"Permission error: {e}")
            continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Extract relevant features
        if 'timestamp' not in data.columns or 'subcarrier' not in data.columns or 'amplitude' not in data.columns or 'phase' not in data.columns:
            print(f"Missing required columns in file: {file}")
            continue

        features = data.loc[:, ['timestamp', 'subcarrier', 'amplitude', 'phase']]

        # Fix timestamp format
        features.loc[:, 'timestamp'] = features['timestamp'].str.replace(':', '-', 2)
        features.loc[:, 'timestamp'] = features['timestamp'].str.replace(':', '.', 1)

        # Convert timestamp to a datetime object
        try:
            features.loc[:, 'timestamp'] = pd.to_datetime(features['timestamp'], format='%Y-%m-%d %H-%M-%S.%f')
        except Exception as e:
            print(f"Error parsing timestamp in file {file}: {e}")
            continue

        # Convert amplitude and phase to numeric values if they are strings
        try:
            if features['amplitude'].dtype == 'object':
                features.loc[:, 'amplitude'] = features['amplitude'].str.replace('.', '').astype(float)
            if features['phase'].dtype == 'object':
                features.loc[:, 'phase'] = features['phase'].str.replace('.', '').astype(float)
        except Exception as e:
            print(f"Error converting amplitude/phase to numeric in file {file}: {e}")
            continue

        # Pivot the data to have subcarriers as columns and timestamps as rows
        try:
            amplitude_pivot = features.pivot(index='timestamp', columns='subcarrier', values='amplitude')
            phase_pivot = features.pivot(index='timestamp', columns='subcarrier', values='phase')
        except Exception as e:
            print(f"Error pivoting data in file {file}: {e}")
            continue

        # Fill any missing values if necessary
        amplitude_pivot.fillna(method='ffill', inplace=True)
        phase_pivot.fillna(method='ffill', inplace=True)

        # Combine amplitude and phase data
        combined_data = pd.concat([amplitude_pivot, phase_pivot], axis=1)
        combined_data = normalize_data(combined_data)  # Normalize data
        data_list.append(combined_data)

    if not data_list:
        raise ValueError("No valid data segments found after preprocessing.")

    all_data = pd.concat(data_list)

    return all_data

if __name__ == '__main__':
    # Load test data
    test_data_list = []

    data_dir = "data/Data/test/*.txt"
    files = glob.glob(data_dir)
    if not files:
        raise ValueError(f"No files found in directory: {data_dir}")

    test_data = load_and_preprocess_data(files)
    test_data_list.append(test_data)

    # Combine test data
    combined_test_data = pd.concat(test_data_list)

    # Ensure all values are numeric
    combined_test_data = combined_test_data.apply(pd.to_numeric, errors='coerce')

    # Debugging output
    print(f"Combined test data shape: {combined_test_data.shape}")
    print(f"Number of test samples: {len(combined_test_data)}")
    print(f"Number of features: {combined_test_data.shape[1]}")

    # Prepare the data for the model
    num_test_samples = len(combined_test_data)
    num_subcarriers = combined_test_data.shape[1] // 2  # Dynamically determine number of subcarriers

    print(f"Number of subcarriers: {num_subcarriers}")

    # Ensure the combined data can be reshaped correctly
    if num_test_samples * 2 * num_subcarriers != combined_test_data.size:
        raise ValueError(f"Cannot reshape data of size {combined_test_data.size} to ({num_test_samples}, 2, {num_subcarriers}, 1)")

    test_tensor = torch.from_numpy(combined_test_data.values).type(torch.FloatTensor).view(num_test_samples, 2, num_subcarriers, 1)

    # Create TensorDataset
    test_dataset = TensorDataset(test_tensor, torch.zeros(num_test_samples, dtype=torch.long))  # Placeholder labels
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    model = ResNetWithDropoutAndBatchNorm(ResidualBlock, [2, 2, 2, 2], num_classes=3).cuda()  # Adjusted for three classes
    model.load_state_dict(torch.load('trained_models/final_model.ckpt'))  # Load the final trained model
    model.eval()

    class_names = ['individual_1', 'random_individual', 'nobody']

    with torch.no_grad():
        for samples, _ in tqdm(test_loader):
            samples = samples.cuda()
            outputs = model(samples)
            softmax_outputs = torch.softmax(outputs, dim=1)
            confidences, predicted = softmax_outputs.max(1)

            for i in range(len(samples)):
                prediction = predicted[i].item()
                confidence = confidences[i].item()
                if confidence < 0.5:
                    label = 'stranger'
                else:
                    label = class_names[prediction]
                print(f'Prediction: {label}, Confidence: {confidence:.2f}')

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
from torch.autograd import Variable
from model.res_net_use_this import ResNetWithDropoutAndBatchNorm, ResidualBlock
from augmentations import apply_augmentations
import utils
from utils import *

# Hyperparameters
batch_size = 64
num_epochs = 100
learning_rate = 1e-5
validation_split = 0.2
patience = 10

# Directory to save the model
save_dir = r'trained_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'final_model.ckpt')

# Function to load and preprocess data
def load_and_preprocess_csv(file_path):
    augmented_data = apply_augmentations(file_path)
    all_augmentations = []
    num_subcarriers = 30

    for key in augmented_data.keys():
        if key == 'sliced' or key == 'reverse_sliced':
            for segment in augmented_data[key]:
                if isinstance(segment, pd.DataFrame):
                    scaler = StandardScaler()
                    try:
                        normalized_data = scaler.fit_transform(segment[['amplitude', 'phase']])
                        num_samples = len(normalized_data) // num_subcarriers
                        if num_samples == 0:
                            continue
                        total_size = num_samples * num_subcarriers
                        reshaped_data = normalized_data[:total_size].reshape(num_samples, 2, 1, num_subcarriers)
                        tensor_data = torch.from_numpy(reshaped_data).type(torch.FloatTensor)
                        all_augmentations.append(tensor_data)
                    except Exception as e:
                        print(f"Skipping segment due to error: {e}")
        else:
            if isinstance(augmented_data[key], pd.DataFrame):
                scaler = StandardScaler()
                try:
                    normalized_data = scaler.fit_transform(augmented_data[key][['amplitude', 'phase']])
                    num_samples = len(normalized_data) // num_subcarriers
                    if num_samples == 0:
                        continue
                    total_size = num_samples * num_subcarriers
                    reshaped_data = normalized_data[:total_size].reshape(num_samples, 2, 1, num_subcarriers)
                    tensor_data = torch.from_numpy(reshaped_data).type(torch.FloatTensor)
                    all_augmentations.append(tensor_data)
                except Exception as e:
                    print(f"Skipping segment due to error: {e}")

    if not all_augmentations:
        raise ValueError("No valid data segments found after augmentation.")

    return torch.cat(all_augmentations)

# Function to load data for individual_1, random_individual, and nobody
def load_data_for_experiment(data_dir):
    individual_1_dir = 'individual_1'
    random_individual_dirs = ['individual_2', 'individual_3', 'individual_4', 'individual_5']  # Add more if needed
    nobody_dir = 'nobody'
    all_data = []
    all_labels = []

    # Load individual_1 data
    individual_1_path = os.path.join(data_dir, individual_1_dir)
    individual_1_files = [os.path.join(individual_1_path, f) for f in os.listdir(individual_1_path) if f.endswith('.txt')]
    for fp in individual_1_files:
        try:
            data = load_and_preprocess_csv(fp)
            all_data.append(data)
            all_labels.append(torch.tensor([0] * data.size(0)))  # Label 0 for individual_1
        except ValueError as e:
            print(f"Skipping file {fp} due to error: {e}")

    # Load random_individual data
    for random_individual_dir in random_individual_dirs:
        random_individual_path = os.path.join(data_dir, random_individual_dir)
        random_individual_files = [os.path.join(random_individual_path, f) for f in os.listdir(random_individual_path) if f.endswith('.txt')]
        for fp in random_individual_files:
            try:
                data = load_and_preprocess_csv(fp)
                all_data.append(data)
                all_labels.append(torch.tensor([1] * data.size(0)))  # Label 1 for random_individual
            except ValueError as e:
                print(f"Skipping file {fp} due to error: {e}")

    # Load nobody data
    nobody_path = os.path.join(data_dir, nobody_dir)
    nobody_files = [os.path.join(nobody_path, f) for f in os.listdir(nobody_path) if f.endswith('.txt')]
    for fp in nobody_files:
        try:
            data = load_and_preprocess_csv(fp)
            all_data.append(data)
            all_labels.append(torch.tensor([2] * data.size(0)))  # Label 2 for nobody
        except ValueError as e:
            print(f"Skipping file {fp} due to error: {e}")

    if not all_data:
        raise ValueError("No valid data files found after processing.")

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)
    return all_data, all_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cprintf(f"Selected torch device is: {device}", 'l_yellow')

# Load data files
data_dir = r'data/Data'
train_data, train_labels = load_data_for_experiment(data_dir)

# Split data into training and validation sets
dataset = TensorDataset(train_data, train_labels)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

model = ResNetWithDropoutAndBatchNorm(ResidualBlock, [2, 2, 2, 2], num_classes=3)  # Adjusted for three classes

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Learning rate scheduler and early stopping
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
early_stopping_patience = patience
best_val_loss = float('inf')
epochs_no_improve = 0

model = nn.DataParallel(model)
model.to(device)
criterion.to(device)

# Training loop with validation and early stopping
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_data_loader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')

    train_accuracy = 100 * correct / total
    print(f'Accuracy of the model on the training data after epoch {epoch + 1}: {train_accuracy:.2f}%')

    # Validation step
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_data_loader:
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_data_loader)
    val_accuracy = 100 * correct / total
    print(f'Accuracy of the model on the validation data after epoch {epoch + 1}: {val_accuracy:.2f}%')

    # Step the scheduler
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), save_path)  # Save the best model
        print(f'Model saved to {save_path}')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping after {epoch + 1} epochs.')
            break

# Save the final model
torch.save(model.state_dict(), save_path)
print(f'Final model saved to {save_path}')

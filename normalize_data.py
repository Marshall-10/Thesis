import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = 'C:\\Users\\Hussein\\OneDrive\\Desktop\\RB Pi\\CSI_Dataset\\Individual_1\\Hussein_24-06-2024_13_34_18.csv'
data = pd.read_csv(file_path, delimiter=';')

# Clean the data
data['amplitude'] = data['amplitude'].astype(str).str.replace('.', '', regex=False).astype(float)
data['phase'] = data['phase'].astype(str).str.replace('.', '', regex=False).astype(float)

# Handle missing values
data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
data.fillna(0, inplace=True)

# Normalize the data
scaler = StandardScaler()
data[['amplitude', 'phase']] = scaler.fit_transform(data[['amplitude', 'phase']])

# Save the normalized data to a new CSV file
normalized_file_path = 'normalized_data.csv'
data.to_csv(normalized_file_path, index=False)

print(f'Normalized data saved to {normalized_file_path}')

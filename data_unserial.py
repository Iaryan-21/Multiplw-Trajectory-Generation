import pickle
import numpy as np
import glob
import os

# Path to the directory containing the files
file_dir = '/Users/aryanmishra/Desktop/DSA/Imagenet16_train/'

# Get a list of all files matching the pattern
file_paths = glob.glob(os.path.join(file_dir, 'train_data_batch_*'))

# Check if any files were found
if not file_paths:
    print("No files found matching the pattern 'train_data_batch_*'")
else:
    print(f"Found {len(file_paths)} files.")

# Initialize a list to hold all the data arrays
all_data = []

# Load and concatenate the data from each file
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        
        # Inspect the keys in the data dictionary
        print(f"Keys in the data dictionary: {data.keys()}")
        
        # Use the correct key 'data'
        if 'data' in data:
            all_data.append(data['data'])
        else:
            print(f"No key 'data' found in file {file_path}")

# Check if data was loaded
if not all_data:
    print("No data loaded from files.")
else:
    # Concatenate all data arrays into a single array
    combined_data = np.vstack(all_data)
    print(f"Combined data shape: {combined_data.shape}")

    # Save the combined data as a CSV file for easy reading in Rust
    output_path = os.path.join(file_dir, 'train_data_combined.csv')
    np.savetxt(output_path, combined_data, delimiter=',')
    print(f"Combined data saved to {output_path}")

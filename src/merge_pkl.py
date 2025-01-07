import os
import pickle

# Base directory containing the chunks
base_dir = "/home/grig0076/susie_scratch/phynteny_transformer/data/PhageScope"

# Prefix for the chunk directories
chunk_prefix = "PhageScope_30122024_representatives_esm2_t33_650M_chunk"

# Dictionary to hold the combined data
merged_data = {}

# Iterate over the 20 directories
for i in range(1, 21):
    chunk_dir = os.path.join(base_dir, f"{chunk_prefix}{i}")
    pickle_file = os.path.join(chunk_dir, "data.X.pkl")
    
    # Load the pickle file
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
            merged_data.update(data)
            print(f"Successfully loaded data from {pickle_file}")
    except Exception as e:
        print(f"Failed to load {pickle_file}: {e}")
    
    # Print progress update
    print(f"Processed {i}/20 directories")

# Save the merged data into a new pickle file
output_file = os.path.join(base_dir, "merged_data.X.pkl")
try:
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)
        print(f"Merged data saved to {output_file}")
except Exception as e:
    print(f"Failed to save merged data: {e}")


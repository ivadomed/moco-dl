import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
CSV_PATH = "data_list.csv"
OUTPUT_DIR = "crop_dataset"
SEED = 42  # Set seed for reproducible 80/20 train/val splits

# Read CSV
df = pd.read_csv(CSV_PATH)

# Clean up empty/NaN values (especially for 'run')
df = df.fillna("")

# Perform 80/20 Train/Validation Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

splits = {
    "train": train_df,
    "validation": val_df
}

for split_name, split_df in splits.items():

    # Create split directory (e.g., ./cropped_dataset/train)
    target_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(target_dir, exist_ok=True)
    
    for idx, row in split_df.iterrows():
        input_path = row["path"]
        
        # Skip row if input file doesn't exist
        if not os.path.isfile(input_path):
            print(f"[WARNING] File not found, skipping: {input_path}")
            continue

        # Extract the original filename from the path
        orig_filename = os.path.basename(input_path)
        
        # Safely insert "_crop" before the extension
        if orig_filename.endswith(".nii.gz"):
            output_filename = orig_filename.replace(".nii.gz", "_crop.nii.gz")
        else:
            name, ext = os.path.splitext(orig_filename)
            output_filename = f"{name}_crop{ext}"
            
        output_path = os.path.join(target_dir, output_filename)
        
        # Build and execute sc_crop command
        cmd = ["sc_crop", "-i", input_path, "-o", output_path]
        
        print(f"[{split_name.upper()}] Processing: {output_filename}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to crop {input_path}: {e}")
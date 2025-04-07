import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# File and directory paths
base_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine"  # Base directory for dataset
filtered_dir = (
    r"C:\Users\bugra\OneDrive\Masaüstü\ML\filtered"  # Directory for filtered images
)
train_dir = os.path.join(base_dir, "training224")  # Directory for training images
val_dir = os.path.join(base_dir, "validation224")  # Directory for validation images
test_dir = os.path.join(base_dir, "test224")  # Directory for test images
csv_file = os.path.join(
    base_dir, "Balanced_Chest_Xray_Data.csv"
)  # Path to the CSV with image data

# Create directories for training, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read CSV and extract relevant columns
df = pd.read_csv(csv_file)[
    ["Image Index", "Finding Labels"]
]  # Load image filenames and their labels

# Split dataset into training (70%), validation (20%), and test (10%) sets
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(
    train_val, test_size=0.2222, random_state=42
)  # 20% of 90% for validation

# Initialize counters and list for missing images
total_train_count, total_val_count, total_test_count = 0, 0, 0
missing_images = []


# Function to resize and save images to the appropriate directory
def process_and_save_images(data, target_dir, count_tracker):
    """Resize and save images to the target directory."""
    os.makedirs(target_dir, exist_ok=True)  # Ensure the target directory exists
    for _, row in data.iterrows():  # Iterate through image data
        src_path = os.path.join(filtered_dir, row["Image Index"])
        dest_path = os.path.join(target_dir, row["Image Index"])

        # Check if the image exists and process it
        if os.path.exists(src_path):
            try:
                with Image.open(src_path) as img:
                    img_resized = img.resize((224, 224))  # Resize image to 224x224
                    img_resized.save(dest_path)  # Save the resized image
                count_tracker += 1  # Increment count for this set
            except Exception as e:
                # Log errors and track missing images
                print(f"Error processing {src_path}: {e}")
                missing_images.append(row["Image Index"])
        else:
            missing_images.append(row["Image Index"])  # Track missing image

    return count_tracker  # Return updated count


# Process and save images for each set: training, validation, and test
total_train_count = process_and_save_images(train, train_dir, total_train_count)
total_val_count = process_and_save_images(val, val_dir, total_val_count)
total_test_count = process_and_save_images(test, test_dir, total_test_count)

# Print statistics about the processed images and any missing images
print(f"Total Training Images (224x224): {total_train_count}")
print(f"Total Validation Images (224x224): {total_val_count}")
print(f"Total Test Images (224x224): {total_test_count}")
print(f"Missing Images: {len(missing_images)}")
if missing_images:
    print("Missing Images List:", missing_images)  # Display list of missing images

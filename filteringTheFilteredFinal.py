import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths for input CSV and directories for images and output data
csv_path = r"C:\Users\bugra\OneDrive\Masaüstü\machine\Balanced_Chest_Xray_Data.csv"
filtered_images_dir = r"C:\Users\bugra\OneDrive\Masaüstü\ML\filtered"
output_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine"

# Read CSV containing image filenames and their labels
data = pd.read_csv(csv_path)

# Ensure the CSV contains necessary columns: 'Image Index' and 'Finding Labels'
if "Image Index" not in data.columns or "Finding Labels" not in data.columns:
    raise ValueError(
        "CSV file must contain 'Image Index' and 'Finding Labels' columns."
    )


# Helper function to create folders if they do not exist
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Create base folders for training, validation, and test datasets
for subset in ["training", "validation", "test"]:
    create_folder(os.path.join(output_dir, subset))

# Group images by class labels ('Finding Labels')
grouped = data.groupby("Finding Labels")

# Initialize counts for the images in each subset
counts = {"training": 0, "validation": 0, "test": 0}

# Initialize a dictionary to track class-wise image counts for each subset
class_counts = {
    subset: {class_name: 0 for class_name in grouped.groups.keys()}
    for subset in ["training", "validation", "test"]
}

# Loop through each class group to process images
for class_name, group in grouped:
    # Get the list of image filenames for this class
    image_filenames = group["Image Index"].tolist()

    # Split the data: 70% for training, 20% for validation, 10% for testing
    train_val, test = train_test_split(image_filenames, test_size=0.1, random_state=42)
    train, val = train_test_split(
        train_val, test_size=0.222, random_state=42
    )  # 20% of original = ~22.2% of train_val for validation

    # Map subsets to their respective image lists
    subsets = {"training": train, "validation": val, "test": test}

    # Copy files to their respective folders based on the subset
    for subset, filenames in subsets.items():
        for filename in filenames:
            src_path = os.path.join(filtered_images_dir, filename)
            dest_folder = os.path.join(output_dir, subset, class_name)
            create_folder(
                dest_folder
            )  # Create folder for the class if it doesn't exist
            dest_path = os.path.join(dest_folder, filename)

            # If source image exists, copy it to the destination
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                counts[subset] += 1  # Increment count for this subset
                class_counts[subset][
                    class_name
                ] += 1  # Increment class count for this subset

# Calculate and print out the summary of images in each subset
total_images = len(data)  # Total number of images in the original dataset
print("Summary:")
for subset in ["training", "validation", "test"]:
    percentage = (
        counts[subset] / total_images
    ) * 100  # Percentage of total images in this subset
    print(
        f"{subset.capitalize()}: {counts[subset]} images ({percentage:.2f}% of total)"
    )
    # Print class-wise distribution for this subset
    for class_name, count in class_counts[subset].items():
        class_percentage = (count / counts[subset]) * 100 if counts[subset] > 0 else 0
        print(f"  {class_name}: {count} images ({class_percentage:.2f}%)")

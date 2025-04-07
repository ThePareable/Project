import os
import shutil
import pandas as pd

# Paths for source images, destination directory, and the CSV file
images_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine"
filtered_dir = r"C:\Users\bugra\OneDrive\Masaüstü\ML\filtered"
csv_file = r"C:\Users\bugra\OneDrive\Masaüstü\ML\Balanced_Chest_Xray_Data.csv"

# Ensure destination directory exists
os.makedirs(filtered_dir, exist_ok=True)

# Load image names from CSV
try:
    df = pd.read_csv(csv_file)
    image_names = df["Image Index"].tolist()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Copy images to the destination directory
failed_images = []  # Track failures
filtered_count = 0  # Track success count

for image_name in image_names:
    src_path = os.path.join(images_dir, image_name)  # Source path
    dest_path = os.path.join(filtered_dir, image_name)  # Destination path

    if os.path.isfile(src_path):  # Check if file exists
        try:
            shutil.copy(src_path, dest_path)  # Copy file
            filtered_count += 1
        except Exception as e:
            print(f"Failed to copy {image_name}: {e}")
            failed_images.append(image_name)  # Log failed copies
    else:
        print(f"Image not found: {image_name}")
        failed_images.append(image_name)  # Log missing files

# Summary of results
print(f"\nFiltering complete!")
print(f"Total images filtered: {filtered_count}")
print(f"Total failures: {len(failed_images)}")

if failed_images:  # Print failed images, if any
    print("\nFailed images:")
    for img in failed_images:
        print(f"- {img}")

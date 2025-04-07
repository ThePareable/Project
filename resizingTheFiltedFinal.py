import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# Define paths for the main directory, filtered images, and output folders
base_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine"
filtered_dir = r"C:\Users\bugra\OneDrive\Masaüstü\ML\filtered"
train_dir = os.path.join(base_dir, "training224")  # Directory for training images
val_dir = os.path.join(base_dir, "validation224")  # Directory for validation images
test_dir = os.path.join(base_dir, "test224")  # Directory for test images
csv_file = os.path.join(base_dir, "Balanced_Chest_Xray_Data.csv")

# Create directories for training, validation, and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load the CSV and select required columns
df = pd.read_csv(csv_file)
df = df[["Image Index", "Finding Labels"]]  # Keep image names and class labels

# Split data into training (70%), validation (20%), and test (10%)
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(
    train_val, test_size=0.2222, random_state=42
)  # 0.2222 ensures 20% of the total

# Initialize counters and a list for missing images
total_train_count, total_val_count, total_test_count = 0, 0, 0
missing_images = []


def create_class_subfolders(base_dir, class_labels):
    """
    Create subfolders for each class label in the specified directory.
    Each class gets its own folder.
    """
    for class_label in class_labels:
        class_dir = os.path.join(base_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)


def process_and_save_images(data, base_dir, count_tracker):
    """
    Resize images to 224x224 and save them to the appropriate class subfolder.
    If an image is missing or errors occur, log it in `missing_images`.
    """
    global missing_images  # Access global list of missing images
    class_labels = data["Finding Labels"].unique()  # Get unique class labels
    create_class_subfolders(base_dir, class_labels)  # Create subfolders for each class

    for _, row in data.iterrows():
        src_path = os.path.join(
            filtered_dir, row["Image Index"]
        )  # Path to the source image
        dest_dir = os.path.join(
            base_dir, row["Finding Labels"]
        )  # Destination subfolder
        dest_path = os.path.join(
            dest_dir, row["Image Index"]
        )  # Path to save resized image

        if os.path.exists(src_path):  # Check if the image file exists
            try:
                # Resize the image to 224x224 and save it to the destination
                with Image.open(src_path) as img:
                    img_resized = img.resize((224, 224))
                    img_resized.save(dest_path)
                count_tracker += 1
            except Exception as e:  # Handle errors during processing
                print(f"Error processing {src_path}: {e}")
                missing_images.append(row["Image Index"])
        else:
            missing_images.append(row["Image Index"])  # Log missing images
    return count_tracker


# Process the datasets and count the resized images
total_train_count = process_and_save_images(train, train_dir, total_train_count)
total_val_count = process_and_save_images(val, val_dir, total_val_count)
total_test_count = process_and_save_images(test, test_dir, total_test_count)

# Display final statistics
print(f"Total Training Images (224x224): {total_train_count}")
print(f"Total Validation Images (224x224): {total_val_count}")
print(f"Total Test Images (224x224): {total_test_count}")
print(f"Missing Images: {len(missing_images)}")
if missing_images:
    print("Missing Images List:", missing_images)

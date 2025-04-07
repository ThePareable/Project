import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_CLASSES = 6  # Number of unique classes in your dataset
val_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine\test224"  # Directory containing validation data
IMAGE_SIZE = (224, 224)  # Image dimensions for resizing
BATCH_SIZE = 32  # Batch size for data processing

# Load Validation Data
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)  # Rescale pixel values to [0, 1]
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,  # Resize images to match the input size of the model
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Multi-class classification
    shuffle=False,  # Maintain file order to match predictions with ground truth
)

# Load the Trained Model
model = tf.keras.models.load_model("best_resnet_model.h5")  # Load the pre-trained model

# Predict on Validation Set
y_true = val_generator.classes  # Ground truth labels from the validation generator
val_steps = (
    val_generator.samples // val_generator.batch_size
)  # Calculate number of steps per epoch
y_pred_proba = model.predict(
    val_generator, steps=val_steps + 1
)  # Predict class probabilities
y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class predictions

# Classification Report
print("Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=val_generator.class_indices.keys(),  # Map indices to class names
    )
)

# ROC-AUC Score
roc_auc = roc_auc_score(
    tf.keras.utils.to_categorical(
        y_true, NUM_CLASSES
    ),  # Convert labels to one-hot encoding
    y_pred_proba,  # Predicted probabilities
    multi_class="ovr",  # One-vs-Rest strategy for multi-class ROC
)
print(f"ROC-AUC Score: {roc_auc:.4f}")  # Display the ROC-AUC score

# Plot ROC Curve
fpr, tpr, _ = roc_curve(
    tf.keras.utils.to_categorical(y_true, NUM_CLASSES).ravel(),  # Flatten true labels
    y_pred_proba.ravel(),  # Flatten predicted probabilities
)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")  # Plot ROC curve
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()  # Display the ROC curve

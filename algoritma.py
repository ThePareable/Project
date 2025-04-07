import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Parameters for model training and setup
IMAGE_SIZE = (224, 224)  # Image size (224x224 is standard for ResNet50)
BATCH_SIZE = 8  # Number of samples processed before updating model weights
EPOCHS = 30  # Number of times to iterate over the entire dataset
NUM_CLASSES = 6  # Number of classes in your classification problem
LEARNING_RATE = 1e-5  # A small learning rate for fine-tuning the model

# Directory paths for training and validation data
train_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine\training224"  # Directory for training data
val_dir = r"C:\Users\bugra\OneDrive\Masaüstü\machine\validation224"  # Directory for validation data

# Augmentation and rescaling for training images to prevent overfitting
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values to [0, 1]
    rotation_range=30,  # Random rotation between -30 and +30 degrees
    width_shift_range=0.2,  # Random horizontal shifts up to 20%
    height_shift_range=0.2,  # Random vertical shifts up to 20%
    shear_range=0.2,  # Random shearing transformations
    zoom_range=0.3,  # Random zooming up to 30%
    horizontal_flip=False,  # Random horizontal flipping (disabled here)
)

# Validation data is only rescaled, no augmentation to preserve integrity
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Create generators that will flow the images and their labels from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,  # Resize images to match model input size
    batch_size=BATCH_SIZE,  # Set the batch size
    class_mode="categorical",  # Categorical labels (for multi-class classification)
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,  # Resize images to match model input size
    batch_size=BATCH_SIZE,  # Set the batch size
    class_mode="categorical",  # Categorical labels
)

# Compute class weights to handle class imbalance in the dataset
class_weights = compute_class_weight(
    "balanced",  # Automatically balance the weights inversely proportional to class frequency
    classes=np.unique(train_generator.classes),  # Unique class labels
    y=train_generator.classes,  # The classes for each sample in the training data
)
class_weights = dict(
    enumerate(class_weights)
)  # Convert the class weights to a dictionary

# Build the model using ResNet50 as a feature extractor (pre-trained on ImageNet)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# The base ResNet50 model without the fully connected layers (include_top=False)

# Add a global average pooling layer to reduce the dimensionality of the feature map
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling to get the average of all spatial features
x = Dropout(0.5)(x)  # Add dropout to avoid overfitting (50% dropout rate)
output = Dense(NUM_CLASSES, activation="softmax")(
    x
)  # Final dense layer for classification with softmax activation

# Construct the full model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with a lower learning rate and label smoothing
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),  # Adam optimizer with a small learning rate
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),  # Label smoothing for better generalization and reduced overconfidence
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
    ],  # Track accuracy and AUC (Area Under the Curve)
)

# Callbacks to improve training
checkpoint = ModelCheckpoint(
    "best_resnet_model.h5",  # Save the model with the best validation accuracy
    monitor="val_accuracy",  # Monitor validation accuracy
    save_best_only=True,  # Save only the best model (highest accuracy)
    mode="max",  # Maximize the validation accuracy
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=10, verbose=1
)  # Stop training if accuracy doesn't improve for 10 epochs
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1
)  # Reduce learning rate if validation loss plateaus

# Train the model with the training and validation data
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,  # Use class weights to balance the classes
    callbacks=[  # Include callbacks for monitoring and adjusting training
        checkpoint,
        early_stopping,
        reduce_lr,
    ],
)

# Evaluate the trained model on the validation set
model.evaluate(
    val_generator
)  # Evaluate the model's performance on the validation dataset

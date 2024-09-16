# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
Develop a model that can classify images of handwritten digits (0-9) from the MNIST dataset with high accuracy. The model should use a convolutional neural network architecture and be optimized using early stopping to avoid overfitting.

## Neural Network Model
![exp2 img DL](https://github.com/user-attachments/assets/d9dd1cb8-8bad-41f0-8704-14b179e79a03)

## DESIGN STEPS

### STEP 1:
Import the necessary packages.

### STEP 2:
Load the dataset and inspect the shape of the dataset.

### STEP 3:
Reshape and normalize the images.

## STEP 4:
Use EarlyStoppingCallback function.

## STEP 5:
Get the summary of the model.


## PROGRAM
### Name:RENUGA S
### Register Number:212222230118
```
import numpy as np
import tensorflow as tf

# Provide path to get the full path
data_path ='/content/mnist.npz'

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

#reshape_and_normalize

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(-1, 28, 28, 1)

    # Normalize pixel values
    images = images / 255.0

    ### END CODE HERE ###

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name:RENUGA S           RegisterNumber: 212222230118         \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

# EarlyStoppingCallback

### START CODE HERE ###

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
     def on_epoch_end(self, epoch, logs=None):

        # Check if the accuracy is greater or equal to 0.995
         val_accuracy = logs.get('val_accuracy')
         if val_accuracy is not None:

            # Stop training once the above condition is met
                if val_accuracy >= self.target_accuracy:
                  self.model.stop_training = True
                  print("\nReached 99.5% accuracy so cancelling training!\n")
                  print('Name: RENUGA S           Register Number:   212222230118      \n')
### END CODE HERE ###

# convolutional_model

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([
		# First convolutional layer with 32 filters and a kernel size of 3x3, followed by a ReLU activation function
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

        # Max pooling layer with pool size 2x2
        tf.keras.layers.MaxPooling2D(2, 2),

        # Second convolutional layer with 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # Max pooling layer
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten the feature maps to feed into the dense layers
        tf.keras.layers.Flatten(),

        # Fully connected (dense) layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),

        # Output layer with 10 units (for 10 classes) and softmax activation for classification
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

    return model

# Define your compiled (but untrained) model
model = convolutional_model()

# Train your model (this can take up to 5 minutes)
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])

model.summary()
```
## OUTPUT

### Reshape and Normalize output
![output1](https://github.com/user-attachments/assets/63515344-c9b8-4225-aead-e2e927865f34)

### Training the model output
![output2](https://github.com/user-attachments/assets/119c4d60-c32a-4962-8842-937c60f39763)

### Model summary:
![output3](https://github.com/user-attachments/assets/6387d94e-f259-439c-ae5f-a1c6706cad4f)

## RESULT:
Thus the program to create a Convolution Neural Network to classify images is successfully implemented.


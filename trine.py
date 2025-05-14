import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom-in
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill in missing pixels after transformations
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Define the directories for training and validation data
train_dir = './dataset_splite/train'
valid_dir = './dataset_splite/valid'

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='categorical'  # Since it's multi-class classification
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Step 2: Build the CNN Model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

# 2nd Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 3rd Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # To prevent overfitting
model.add(Dense(3, activation='softmax'))  # 3 classes: lung_aca, lung_scc, lung_n

# Step 3: Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Since it's multi-class classification
    metrics=['accuracy']
)

# Step 4: Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

# Step 5: Evaluate the Model
test_dir = './dataset_splite/train'
test_generator = valid_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Step 6: Visualize the Training Results
# Plotting the training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Plotting the training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# Step 7: Save the Model
model.save('lung_cancer_model.h5')

# Step 8: Make Predictions (Example)
from tensorflow.keras.preprocessing import image

img_path = './lungaca1.jpeg'
img = image.load_img(img_path, target_size=(150, 150))

# Preprocess the image for the model
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make predictions
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Map the predicted class index to the class label
class_labels = ['lung_aca', 'lung_scc', 'lung_n']
print(f"The predicted class is: {class_labels[predicted_class]}")



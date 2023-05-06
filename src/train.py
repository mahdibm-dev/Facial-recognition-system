import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from model import build_model

# Set the input image shape and number of classes
input_shape = (48, 48, 1)
num_classes = 7

# Define the model architecture
model = build_model(input_shape, num_classes)

# Print the model summary
model.summary()

# Compile the model with the categorical cross-entropy loss function and the Adam optimizer
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=input_shape[:2],
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=input_shape[:2],
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

# Train the model on the data generators for a specified number of epochs
epochs = 50

history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples/val_generator.batch_size)

#

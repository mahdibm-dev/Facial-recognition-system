import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Set the input image shape and number of classes
input_shape = (48, 48, 1)
num_classes = 7

# Load the trained model
model = load_model('model.h5')

# Define a dictionary to map the emotion labels to their corresponding index in the model's output
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load and preprocess the input image
img_path = 'test_image.jpg'
img = load_img(img_path, color_mode='grayscale', target_size=input_shape[:2])
img_arr = img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr /= 255.

# Use the model to predict the emotion label of the input image
predictions = model.predict(img_arr)
emotion_index = np.argmax(predictions)
emotion_label = emotion_labels[emotion_index]
confidence = predictions[0][emotion_index]

# Print the predicted emotion label and confidence
print(f'Emotion: {emotion_label} ({confidence:.2f})')

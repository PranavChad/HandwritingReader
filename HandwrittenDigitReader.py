'''
Pranav Chadalavada 1/8/24
This program implements a neural network for handwritten digit recognition using
the MNIST dataset. It preprocesses the data, defines a neural network
architecture with two hidden layers and an output layer, trains the model, and
saves it. The program then allows the user to input an image file, preprocesses
the image, and uses the trained model to predict the digit, displaying the
result.
'''
#Import all necessary libraries for data analysis, image processing
import cv2
import numpy as np
import tensorflow as tf #used for image recog
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from PIL import Image

# Loads in dataset of handwritten digits from tensorflow
mnist = tf.keras.datasets.mnist

# Start Preprocessing
#x_train: training set of images, y_train: corresponding labels for training
#x_test: test set of images, y_test: corresponding labels for testing
(x_train, y_train), (x_test, y_test) = mnist.load_data() #load_data function splits into training and testing data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# End Preprocessing

# Start Neural Network
# Define the input layer
input_layer = tf.keras.layers.Input(shape=(28, 28))
# Flatten the input layer
flattened = tf.keras.layers.Flatten()(input_layer)
# First hidden layer
hidden1 = tf.keras.layers.Dense(128, activation='relu')(flattened)
# Second hidden layer
hidden2 = tf.keras.layers.Dense(128, activation='relu')(hidden1)
# Output layer
output_layer = tf.keras.layers.Dense(10, activation='softmax')(hidden2)
# Create the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Train Model
model.fit(x_train, y_train, epochs = 3)
model.save('handwritten.model')
#End Neural Network

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

#ask user for which file they want to read
print("This neural network has a loss of ", loss, " and an accuracy of ", accuracy)
val = input("Which file do you want to read?: ")
path = '/content/drive/MyDrive/' + val
try:
  img = Image.open(path) #opens filepath
  img_arr = np.invert(np.array(img.convert('L'))) #converts image to grayscale
  img_arr = img_arr / 255.0 #nromalizes pixel values of image to be in range 0-1
  img_arr = cv2.resize(img_arr[0], (28,28)) #resizes images to a shape of (28,28)
  img_arr = np.expand_dims(img_arr, axis=0)
  output = model.predict(img_arr) #predicts digit in input image and stores in output

  #Display the prediction
  print(f"This is {np.argmax(output)}")
except:
  print("Error!")
finally:
  print("End of Program")

import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import IPython.display as display
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



batch_size = 128
epochs = 15
IMG_HEIGHT = 100
IMG_WIDTH = 100

import glob
path = "/Users/Digital/PycharmProjects/ML_Flyzer/hr/Data/"
train_dir = path + "Training"
test_dir = path + "Test"
#os.mkdir("/Users/Digital/PycharmProjects/ML_Flyzer/")
train_dict = {}
test_dict = {}



train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_directories = (glob.glob(train_dir + "/*/"))

for directory in train_directories:
    train_dict[directory.split("\\")[1]] =  glob.glob(directory + "/*")

print(train_dict)

test_directories = (glob.glob(test_dir + "/*/"))

for directory in test_directories:
    test_dict[directory.split("\\")[1]] =  glob.glob(directory + "/*")

print(test_dict)





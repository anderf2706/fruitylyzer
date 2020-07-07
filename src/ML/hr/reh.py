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


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

batch_size = 50
epochs = 15
IMG_HEIGHT = 100
IMG_WIDTH = 100

import glob
path = os.getcwd() + "\\Data\\"
train_dir = path + "Training"
test_dir = path + "Test"
#os.mkdir("/Users/Digital/PycharmProjects/ML_Flyzer/")
train_dict = {}
test_dict = {}
print(train_dir)


def makeGenerators():
    global train_image_generator
    global validation_image_generator
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

makeGenerators()




def readpicturetoDict():
    train_directories = (glob.glob(train_dir + "/*/"))

    for directory in train_directories:
        train_dict[directory.split("\\")[10]] = glob.glob(directory + "/*")

    test_directories = (glob.glob(test_dir + "/*/"))

    for directory in test_directories:
        test_dict[directory.split("\\")[10]] = glob.glob(directory + "/*")


readpicturetoDict()

def checklength(dict):
    val = 0
    for lists in dict.values():
        val += len(lists)
    print(val)

checklength(test_dict)

def prepareData(train, test):
    global train_data_gen
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    global test_data_gen
    test_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=test_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    """
    global dataset_training
    dataset_training = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    global dataset_test
    dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
        test,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    """


prepareData(train_dir, test_dir)

sample_training_images, _ = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:10])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    #Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=67692 // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps=22688 // batch_size
)





acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()





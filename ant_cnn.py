from os import path
import os
import time
import random
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model

import numpy as np
import pandas as pd

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

########################################################################################################################

def build_model():
    model = Sequential(
        [
            Input(shape=(100, 100, 1)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1, activation="softmax")
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

########################################################################################################################

def train_model(model, train_data, test_data, train_labels, test_labels):
    model.fit(train_data, train_labels, batch_size=128, epochs=5, validation_split=0.1)

    predictions = model.predict(test_data)
    print(predictions)
    score = model.evaluate(test_data, test_labels, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

########################################################################################################################

if __name__ == '__main__':
    truth_directory = "images/truths"
    false_directory = "images/falses"

    false_images = []
    false_labels = []
    true_images = []
    true_labels = []




    for filename in os.listdir(truth_directory):
        try:
            image = cv2.imread(truth_directory + "/" + filename, cv2.IMREAD_GRAYSCALE)
            assert image.shape == (100, 100), "img %s has shape %r" % (truth_directory + "/" + filename, image.shape)
            true_images.append(image)
            true_labels.append(1)
        except:
            continue

    for filename in os.listdir(false_directory):
        try:
            image = cv2.imread(truth_directory + "/" + filename, cv2.IMREAD_GRAYSCALE)
            assert image.shape == (100, 100), "img %s has shape %r" % (truth_directory + "/" + filename, image.shape)
            false_images.append(image)
            false_labels.append(0)
        except:
            continue

    length = min(len(true_images), len(false_images))
    true_data = list(zip(true_images, true_labels))
    false_data = list(zip(false_images, false_labels))
    data = random.sample(true_data, length) + random.sample(false_data, length)

    #images, labels =

    #images = np.array(images)
    #data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = zip(*data)

    images = np.array(images)
    print(images.shape)
    labels = np.array(labels)

    # Normalize images
    images = images / 255.0
    images = np.expand_dims(images, -1)

    split_point = int(len(images)*0.8)
    train_images = images[:split_point]
    train_labels = labels[:split_point]
    test_images = images[split_point:]
    test_labels = images[split_point:]

    model = build_model()
    train_model(model, train_images, test_images, train_labels, test_labels)

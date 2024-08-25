import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
urn model

def Model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (2, 2), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(9, activation="softmax")
    ])

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('final_digit_recog_iter1.weights.h5')
    return model

def Predd(squaresh):
    model = Model()
    pred_square = []
    pred = []
    pred_max = []
    for i in range(len(squaresh)):
        pred_square.append(cv2.resize(squaresh[i],(32,32)))
    pred_square = np.array(pred_square)
    for img in pred_square:
        img = img.reshape((1,) + img.shape)
        arre = model.predict(img,verbose=0)
        if np.max(arre)>0.8:
            pred.append(np.argmax(arre)+1)
            pred_max.append(np.max(arre))
        else:
            pred.append(0)
            pred_max.append(np.round(np.max(arre),2))
    return np.array(pred).reshape(9,9)

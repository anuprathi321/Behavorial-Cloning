#!/usr/bin/env python

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

#read data from csv
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#read center, left and right image. Augument dataset by flipping image and negating angle.
images = []
measurements = []
try:
    for line in lines:
        #center image
        source_path=line[0]
        file_name=source_path.split('/')[-1]
        source_path='./data/IMG/'+file_name
        image=cv2.imread(source_path)

        images.append(image)
        image_flipped=np.fliplr(image)
        images.append(image_flipped)

        measurement=float(line[3]) 
        measurements.append(measurement)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)

        #left image
        source_path = line[1]
        file_name = source_path.split('/')[-1]
        source_path = './data/IMG/' + file_name
        image = cv2.imread(source_path)
        images.append(image)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement = float(line[3]) + 0.2
        measurements.append(measurement )
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)

        #right image
        source_path = line[2]
        file_name = source_path.split('/')[-1]
        source_path = './data/IMG/' + file_name
        image = cv2.imread(source_path)
        images.append(image)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)

        measurement = float(line[3]) - 0.2
        measurements.append(measurement)
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)

except:
    Raise("error while reading")

#Augument dataset by adding random brightness. 
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#convert images and measurements into numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

print(X_train[0].shape)
print(y_train[0])
cv2.imshow('Image', images[0])

#generator for training dataset which loads dataset in memory in batches. Else CUDA_OUT_OF_MEMORY ERROR is seen.
#Training dataset is augumented by adding random brightness
def generator_training(features, labels, batch_size=128):
    total_size=len(labels)
    while True:
        shuffle(features, labels)
        for offset in range(0, total_size - batch_size, batch_size):
            end = offset + batch_size
            #labels_corrected = labels[offset:end]*(1+np.random.uniform(-0.10, 0.10))
            images_c=[]
            labels_c=[]
            for i in range(batch_size):
                images_c.append(random_brightness(features[offset + i]))
                labels_c.append(labels[offset + i]*(1+np.random.uniform(-0.10, 0.10)))
            images_c = np.array(images_c)
            labels_c = np.array(labels_c)
            yield (images_c, labels_c)

#gnerator for validation dataset
def generator_validation(features, labels, batch_size=128):
    total_size=len(labels)
    while True:
        shuffle(features, labels)
        for offset in range(0, total_size, batch_size):
            end = offset + batch_size
            yield (features[offset:end], labels[offset:end])

#split train and validation dataset
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

#Model is based on 'Nvidia's End to End Learning for Self-Driving cars' paper.
def getModel():

    model = Sequential()
    model.add(Lambda(lambda x:x /255.0 - 0.5, input_shape=(160, 320, 3)))
    
    #crop top 70 pixels which contains sceneray and bottom 20 pixels which remove car front panel.
    model.add(Cropping2D(cropping=((70,20),(0,0))))

    #Add 3,1,1 convolutional layer to help model learn correct color scheme. Conversion of RGB->YUV was suggested in Nvidia's paper.
    model.add(Convolution2D(3,1,1, subsample=(1,1)))
    model.add(Activation("linear"))

    model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
    model.add(Activation("relu"))

    model.add(Convolution2D(36,5,5, subsample=(2,2)))
    model.add(Activation("relu"))

    model.add(Convolution2D(48,5,5, subsample=(2,2)))
    model.add(Activation("relu"))

    model.add(Convolution2D(64,3,3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Convolution2D(64,3,3))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1164))
    model.add(BatchNormalization())

    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(BatchNormalization())

    model.add(Dense(50))

    model.add(Dropout(0.3))
    model.add(Dense(10))

    model.add(Dense(1))
    return model

model = getModel()

train_generator = generator_training(X_train, y_train, 128)
valid_genetator = generator_validation(X_valid, y_valid, 128)

model.compile(optimizer='adam', loss='mse')
# fir model without generator
#model.fit(X_train, y_train, validation_split=0.2, nb_epoch=2, shuffle=True )

#fit model using generetors
history = model.fit_generator(train_generator, steps_per_epoch=len(y_train)/128, validation_data=valid_genetator, validation_steps=len(y_valid)/128, epochs=15, verbose=1)

#save model
model.save('model.h5')
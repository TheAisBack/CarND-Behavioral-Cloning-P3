import os
import csv
import cv2
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            center = 0
            left = 1
            right = 2 
            measurement = 0.25
            # loop through for left, right and center.
            for clr in range(2):
                for batch_sample in batch_samples:
                    # clr is plugged into for center, right & left.
                    name = './data/IMG/' + batch_sample[clr].split('/')[-1]
                    clr_image = cv2.imread(name)

                    clr_angle = float(batch_sample[3])
                    # center no adjustments
                    if clr == center:
                        clr_angle == measurement
                        angles.append(clr_angle)
                    # if left, trigger an extra adjustment of .25
                    if clr == left:
                        clr_angle += measurement
                        angles.append(clr_angle)
                    # if right, trigger an negative extra adjustment of .25
                    if clr == right:
                        clr_angle -= measurement
                        angles.append(clr_angle)
                        
                    images.append(clr_image)

                    # use augmented measurements for flipping the images.
                    augmented_image = cv2.flip(clr_image, 1)
                    augmented_measurement = clr_angle * -1.0
                    images.append(augmented_image)
                    angles.append(augmented_measurement)

            # trimming the image to only see sections with road and not sky
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch = 160, 320, 3 # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', lr=0.0001) #'mae' 'mse' Learning rate adjustments to 0.0001, better to use 0.0001 to get rid of useless data.

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=25)
# Save model
model.save('model.h5')
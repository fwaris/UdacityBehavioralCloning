import csv
import cv2
import numpy as np
folder = "C:/Users/fwaris/Documents/ws_data/"

def readdata():
    lines = []
    with open(folder + "driving_log.csv") as f:
        for l in f:
            l = l.replace('\\','/')
            elems = l.split(',')
            lines.append(elems)
    return lines

#lines[0]

def readimages(lines):
    images = []
    measurements = []
    for l in lines:
        sp = l[0]
        fn = sp.split('/')[-1]
        p = folder + 'IMG/'  + fn
        image = cv2.imread(p)
        images.append(image)
        measurements.append(l[3])
    return np.array(images),np.array(measurements)

lines = readdata()
X_train,y_train = readimages(lines)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
shape = np.shape(X_train[0])
model = Sequential()

model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=shape))
model.add(Cropping2D(70,25),(0,0))
model.add(Flatten(input_shape=shape))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, epochs=20)
model.save('model.h5')


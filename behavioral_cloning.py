import csv
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO

folder = "C:/Users/fwaris/Documents/ws_data/"
#folder = "F:/fwaris/data/ws_data/"

#read th
def readdata():
    lines = []
    with open(folder + "driving_log.csv") as f:
        for l in f:
            l = l.replace('\\','/')
            elems = l.split(',')
            lines.append(elems)
    return lines

#lines[0]

def loadImage (sp) :
    fn = sp.split('/')[-1]
    p = folder + 'IMG/'  + fn
    image = Image.open(p)
    return np.asarray(image)
  
def readimages(lines):
    images = []
    measurements = []
    for l in lines:
        correction = 0.2
        swa_c = float(l[3])
        swa_l = swa_c + correction
        swa_r = swa_c - correction
        measurements.extend([swa_c,swa_l,swa_r])
        rowImages = map(loadImage,[l[0],l[1],l[2]])
        images.extend(rowImages)
    return np.array(images),np.array(measurements)

lines = readdata()
X_train,y_train = readimages(lines)
shape = np.shape(X_train[0])

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.utils import plot_model
model = Sequential()

model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=shape))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu",kernel_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.30))
model.add(Flatten(input_shape=shape))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#adam = Adam(lr=0.01)
adam = Adam()
model.compile(loss='mse', optimizer=adam)
#plot_model(model,to_file='C:/Users/fwaris/Source/Repos/CarND-Behavioral-Cloning-P3/model.png')
cp = model.fit(X_train,y_train,validation_split=0.2,shuffle=True, epochs=15)
model.save('bh.h5')
print(cp.history.keys())

import matplotlib.pyplot as plt
plt.plot(cp.history['loss'])
plt.plot(cp.history['val_loss'])
plt.title('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()

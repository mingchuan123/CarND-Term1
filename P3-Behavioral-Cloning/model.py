
#Image Reading
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images1 = []
images2 = []
images3 = []
measurements1 = []
for line in lines[1:]:
    
    source_path1 = line[0]
    filename1 = source_path1.split('/')[-1]
    current_path1 = 'data/IMG/' + filename1
    
    image1 = cv2.imread(current_path1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    measurement = float(line[3])
    images1.append(image1)
    measurements1.append(measurement)


X_train1 = np.array(images1)
y_train1 = np.array(measurements1)


#Neural Network Architecture
from keras.models import Model,Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Lambda, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as B
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.backend import tf as ktf


model = Sequential()
model.add(Cropping2D(cropping=((60,40),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0-0.5))
model.add(Conv2D(24, (5, 5), activation='elu', subsample=(2,2), init='random_normal', padding='same'))
model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2,2), init='random_normal', padding='same'))
model.add(Conv2D(64, (5, 5), activation='elu', subsample=(2,2), init='random_normal', padding='same'))
model.add(Conv2D(64, (3, 3), activation='elu', init='random_normal', padding='same'))
model.add(Conv2D(64, (3, 3), activation='elu', init='random_normal', padding='same'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1, activation='linear'))
adam = Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-04, decay=0.0)
model.compile(loss='mse',optimizer=adam,metrics=['accuracy'])

model.fit(X_train1,y_train1,validation_split=0.2,shuffle=True,nb_epoch=6)
model.save('model.h6')

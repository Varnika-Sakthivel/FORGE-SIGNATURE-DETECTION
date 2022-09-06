import pandas as pd
import numpy as np
import skimage.io as sk
from skimage import img_as_ubyte
from skimage.io import imread
from scipy import spatial
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, MaxPooling2D, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
image1 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Real/original_10_1.png")
image2 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Real/original_10_10.png")
image3 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Real/original_10_11.png")
image4 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Real/original_10_12.png")
image5 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Real/original_10_13.png")

fig, ax = plt.subplots(1,5, figsize = (15,10))

ax[0].imshow(image1)
ax[0].set_title("Real_10")
ax[1].imshow(image2)
ax[1].set_title("Real_10")
ax[2].imshow(image3)
ax[2].set_title("Real_10")
ax[3].imshow(image4)
ax[3].set_title("Real_10")
ax[4].imshow(image5)
ax[4].set_title("Real_10")

image6 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Forge/forgeries_10_1.png")
image7 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Forge/forgeries_10_10.png")
image8 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Forge/forgeries_10_11.png")
image9 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Forge/forgeries_10_12.png")
image10 = sk.imread("/content/drive/My Drive/Real-Forg-Signature/Train/Forge/forgeries_10_13.png")

fig, ax1 = plt.subplots(1,5, figsize = (15,10))

ax1[0].imshow(image6)
ax1[0].set_title("Forge_10")
ax1[1].imshow(image7)
ax1[1].set_title("Forge_10")
ax1[2].imshow(image8)
ax1[2].set_title("Forge_10")
ax1[3].imshow(image9)
ax1[3].set_title("Forge_10")
ax1[4].imshow(image10)
ax1[4].set_title("Forge_10")

train_path = '/content/drive/My Drive/Real-Forg-Signature/Train'
test_path = '/content/drive/My Drive/Real-Forg-Signature/Test'

Image_Width = 512
Image_Height = 512
Image_Size = (Image_Width, Image_Height)
Image_Channel = 3
batch_size=15

model = Sequential()


model.add(Conv2D(32, (3,3), activation='relu', input_shape=(Image_Width,Image_Height, Image_Channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
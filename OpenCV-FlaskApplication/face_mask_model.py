from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

DIRECTORY=r"D:\Dataset"
CATEGORIES=["Type1","Type2","Type3","Type4"]


data=[]
labels=[]
import cv2
for category in CATEGORIES:
    folder=os.path.join(DIRECTORY,category)
    label = CATEGORIES.index(category)
    for image in os.listdir(folder):
        image_path=os.path.join(folder,image)
        image_arr=cv2.imread(image_path)
        image_arr=cv2.resize(image_arr,(224,224))
        data.append([image_arr,label])


X = []
y = []
for features,labels in data:
    X.append(features)
    y.append(labels)


Image = np.array(X)
Image_Labels = np.array(y)


import pickle
Image = pickle.load(open("X.pkl", 'rb'))
Label = pickle.load(open("y.pkl",'rb'))

data = np.array(Image, dtype="float32")
data=data/255

baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = baseModel.output

model= Conv2D(filters=64,
                kernel_size=(3,3),
                activation='relu'
                )(model)
model= MaxPooling2D(pool_size=(2,2))(model)

model = Flatten()(model)
model = Dense(units=128,activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(units=4,activation = 'softmax')(model)


model = Model(inputs=baseModel.input, outputs=model)

for layer in baseModel.layers:
	layer.trainable = False

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(data,Label,epochs=5,batch_size=8,validation_split=0.2)

from tensorflow.keras.preprocessing import image

test_image = image.load_img(r"D:\DATA\Downloads\download.jpg",target_size=(224,224))

test = image.img_to_array(test_image)
test = np.array(test,dtype="float32")
test = test/255.

import numpy as np
#this step is required when you have only 1 image
#adding the dimension to number of rows
test = np.expand_dims(test,axis=0)
test.shape

pred=model.predict(test)
print(pred)

import numpy as np
pred = np.argmax(pred,axis=1) #argmax which neuron has the maximum probabilites
print(pred)

model.save("face_mask.model", save_format="h5")
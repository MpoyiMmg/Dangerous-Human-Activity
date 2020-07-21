from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
import cv2
import math
import os
from glob import glob
from scipy import stats as s
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from Model import Model

base_model = VGG16(weights='imagenet', include_top=False)
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.load_weights("weight.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

f = open("dataset_test.csv")
temp = f.read()
videos = temp.split('\n')


test = pd.DataFrame()
test['videos_name'] = videos
test = test[:-1]
test_videos = test['videos_name']
test.head()

train = pd.read_csv('train_new_dataset.csv')
y = train['class']
y = pd.get_dummies(y)

predict = []
actual = []

for i in tqdm(range(test_videos.shape[0])) : 
    count = 0
    videoFile = test_videos[i]
    # print(videoFile)
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5)
    x = 1

    while cap.isOpened :
        frameId = cap.get(1)
        ret, frame = cap.read()

        if ret != True :
            break

        if frameId % math.floor(frameRate) == 0 :
            file_name = 'temp/' + "_frame%d.jpg" % count;count += 1
            cv2.imwrite(file_name, frame)

    cap.release()

    images = glob("temp/*")

    prediction_images = []

    for i in range(len(images)) :
        img = image.load_img(images[i], target_size=(224, 224, 3))
        img = image.img_to_array(img)
        # img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = img/255

        prediction_images.append(img)

    prediction_images = np.array(prediction_images)
    
    prediction_images = base_model.predict(prediction_images)
    print(prediction_images)

    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)

    prediction = model.predict_classes(prediction_images)

    predict.append(y.columns.values[s.mode(prediction)[0][0]])
    actual.append(videoFile.split('/')[2].split('.')[0].split('_')[0])

from sklearn.metrics import accuracy_score
print(accuracy_score(predict, actual) * 100)
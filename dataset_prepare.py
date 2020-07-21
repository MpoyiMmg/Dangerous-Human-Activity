import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import skimage
import cv2
import math

# Read all in the dataset file and get the videos folder and videos names
def open_train_data_set():
    file = open('dataset_train.csv')
    temp = file.read()
    videos = temp.split('\n')
    print(videos)

    return videos

# Read all in the test dataset file and get the videos folder and videos names
def open_validation_data_set():
    file_test = open('dataset_test.csv')
    temp_test = file_test.read()
    videos_test = temp_test.split('\n')

    return videos_test


# creating the dataframe with the videos name
def create_dataframe_of_image_name() : 
    train = pd.DataFrame()
    train['videos_name'] = open_train_data_set()
    train = train[:-1]
    train.head()

    return train


def create_dataframe_of_test_image():
    test = pd.DataFrame()
    test['videos_name'] = open_validation_data_set()
    test = test[:]

    test.head()

    return test

# getting a tag of each video
def get_tag_of_each_video():
    tag_video_name = []
    test_video_tag = []

    train = create_dataframe_of_image_name()
    test = create_dataframe_of_test_image()


    for i in range(train.shape[0]) :
        tag_video_name.append(train['videos_name'][i].split('/')[1])

    for i in range(test.shape[0]) :
        test_video_tag.append(test['videos_name'][i].split('/')[1])

    train['tag'] = tag_video_name
    test['tag'] = test_video_tag

    return train, test

# we'll extract the frame of each image for model training
def get_frame_of_each_video():

    train, _ = get_tag_of_each_video()

    for i in range(train.shape[0]):
        count = 0
        video_file = train['videos_name'][i]
    
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)
        x = 1

        while(cap.isOpened):
            frame_id = cap.get(1)
            ret, frame = cap.read()

            if ret != True:
                break

            if frame_id % math.floor(frame_rate) == 0:
                file_name = 'Dataset/train_test/'+ video_file.split('/')[-1] + 'frame%d.jpg' % count;count += 1
                print(file_name)
                cv2.imwrite(file_name, frame)

        cap.release

# getting the name of images
def get_name_of_images():
    images = glob("Dataset/train_test/*.jpg")
    
    if len(images) == 0 :
        get_frame_of_each_video()
        images = glob("Dataset/train_test/*.jpg")

    train_image = []
    train_class = []

    for i in tqdm(range(len(images))) :
         train_image.append(images[i].split('/')[-1])
         train_class.append(images[i].split('/')[-1].split('_')[0])

    train_data = pd.DataFrame()
    train_data['images'] = train_image
    train_data['class'] = train_class

    # generate a new dataset with images and tags of each
    train_data.to_csv("train_new_dataset.csv", header=True, index=False)
    print("Done!")


if __name__ == "__main__":
    get_name_of_images()
from keras.preprocessing import image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model import Model


def load_data():
    train = pd.read_csv('train_new_dataset.csv')
    train.head()

    return train

def convert_images_to_numpy_array():
    train_image = []
    train = load_data()
    
    for i in tqdm(range(train.shape[0])) :
        img = image.load_img('Dataset/train/' + train['images'][i], target_size=(224, 224, 3))
        
        # convert image into numpy array
        img = image.img_to_array(img)

        # normalise values of pixels
        img = img/255

        train_image.append(img)

    X = np.array(train_image)
    X.shape
    
    return X

def get_dummies():
    train = load_data()
    X = convert_images_to_numpy_array()
    
    y = train['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return X_train, X_test, y_train, y_test

def reshape_train_test_data(X_train, X_test):
    return X_train.reshape(499, 7, 7, 512), X_test.reshape(125, 7, 7, 512)

def train_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    X_train, X_test, y_train, y_test = get_dummies()

    X_train = base_model.predict(X_train)
    X_train.shape

    X_test = base_model.predict(X_test)
    X_test.shape

    # output: (499, 7, 7,7512) : train
    # output: (125, 7 ,7, 512) : test

    # X_train, X_test = reshape_train_test_data(X_train, X_test)
    X_train = X_train.reshape(499, 7*7*512)
    X_test = X_test.reshape(125, 7*7*512)

    # normalisation des valuers des pixels
    max = X_train.max()
    X_train = X_train/max
    X_test = X_test/max

    X_train.shape

    model = Model()
    mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    loss = 'categorical_crossentropy'
    optimizer = 'Adam'
    metrics = 'accuracy'

    model.compile(loss, optimizer, metrics)
    print("Model compilation...")

    model.get_model_summury()
    print('Training model ...')
    
    model.fit_data(X_train, X_test, y_train, y_test, mcp_save)
    print('Done!')   


# launch the training model
if __name__ == "__main__" :
    train_model()
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPool2D
from keras.preprocessing import image


class Model :
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(1024, activation='relu', input_shape=(25088,)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
    
    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    
    def get_model_summury(self):
        return self.model.summary()

    def fit_data(self, X_train, X_test, y_train, y_test, mcp_save):
        self.model.fit(
            X_train,
            y_train,
            epochs=200,
            validation_data=(X_test, y_test),
            callbacks=[mcp_save],
            batch_size=128
        )
    
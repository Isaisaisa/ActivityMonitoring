from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

class MlpClassifier():

    def __init__(self):
        self.num_classes = 55

        # Define model architecture
        self.model = Sequential()
        self.model.add(Dense(400, activation='relu', input_shape=(225,)))
        self.model.add(Dense(400, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        self.model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self, data, labels):
        labels = np_utils.to_categorical(labels, self.num_classes)

        # Fit model on training data
        self.model.fit(data, labels,
                  batch_size=100, epochs=100, verbose=1)


    def eval(self, in_test, out_test):
        # # Evaluate model on test data
        labels = np_utils.to_categorical(out_test, 55)

        score = self.model.evaluate(in_test, labels, verbose=0)
        print('Score acc: ', score[1])
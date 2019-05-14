from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import tensorflow as tf

class MlpClassifier():

    def __init__(self, data, labels):
        #self.data = data
        print(data.shape)
        self.data = data.reshape(1, 1, data.shape[0], data.shape[1])
        self.num_classes = 55
        # self.label = dataLoader.loadTrainingLabels()
        #self.labels = labels
        self.labels = np_utils.to_categorical(labels, 55)
        print(self.data.shape)
        print(self.labels.shape)

        # Define model architecture
        self.model = Sequential()
        self.model.add(Dense(1692, input_shape=(1, 1692, 150)))
        self.model.add(Activation('relu'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(55))
        self.model.add(Activation('softmax'))

        # Compile model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    def train(self):
        # Fit model on training data
        self.model.fit(self.data, self.labels,
                  batch_size=100, epochs=10, verbose=1)


    def eval(self, in_test, out_test):
        # # Evaluate model on test data
        score = self.model.evaluate(in_test, out_test, verbose=0)
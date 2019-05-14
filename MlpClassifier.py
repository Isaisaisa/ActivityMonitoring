from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

class MlpClassifier():

    def __init__(self, data):
        self.data = data
        self.num_classes = 55
        # 4. Load pre-shuffled MNIST data into train and test sets
        #(X_train, y_train), (X_test, y_test) = mnist.load_data()


    def train_and_eval(self):
        # Define model architecture
        model = Sequential()
        print(self.data.shape)
        # model = Sequential()
        # model.add(Dense(512, input_shape=(1,1)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(self.num_classes))
        # model.add(Activation('softmax'))
        #
        # # Compile model
        # model.compile(loss='categorical_crossentropy',
        #               optimizer='adam',
        #               metrics=['accuracy'])
        #
        # # Fit model on training data
        # model.fit(X_train, Y_train,
        #           batch_size=32, nb_epoch=10, verbose=1)
        #
        # # Evaluate model on test data
        # score = model.evaluate(X_test, Y_test, verbose=0)
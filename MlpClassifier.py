from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import np_utils
import numpy as np
from keras import optimizers


class MlpClassifier():

    def __init__(self, inputShape):
        np.random.seed(123)
        self.num_classes = 55

        # Define model architecture
        self.model = Sequential()
        self.model.add(Dense(inputShape, input_shape=(inputShape,)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation='sigmoid'))
        self.model.add(Dropout(0.40))
        self.model.add(Dense(450))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation='sigmoid'))
        self.model.add(Dropout(0.40))
        self.model.add(Dense(450))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation='sigmoid'))
        self.model.add(Dropout(0.40))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        adam = optimizers.Adam(lr=0.0006, beta_1=0.9, beta_2=0.999)
        # Compile model
        # loss function -> accuracy
        # mean_squared_error -> 0.32785923755413626
        # categorical_crossentropy -> 0.3331378299295028
        # mean_absolute_error -> 0.18768328446621768
        # sparse_categorical_crossentropy -> dim error
        # mean_squared_logarithmic_error -> 0.3272727272902066
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

    def train(self, data, labels):
        labels = np_utils.to_categorical(labels, self.num_classes)

        # Fit model on training data
        self.model.fit(data, labels,
                  batch_size=100, epochs=500, verbose=1)

    def eval(self, in_test, out_test):
        # Evaluate model on test data
        labels = np_utils.to_categorical(out_test, 55)

        score = self.model.evaluate(in_test, labels, verbose=0)
        return score[1]

    def predict(self, in_test):
        labels = self.model.predict(in_test)
        return labels

    def predicted_labels(self, in_test):
        labels = self.model.predict(in_test)
        # returns the most likely labels along all ~1700 samples
        return np.argmax(labels, axis=1)
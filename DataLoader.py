## This is the DataLoader
## With this

import numpy as np
import config_dev as config

class DataLoader:

    ## Konstruktor um die Daten zu laden
    def __init__(self):
        ##self.data = []
        self.trainingPath = config.TRAININGFILES
        self.testPath = config.TESTINGFILES
        self.savePath = config.SAVEPATH

    def loadOriginalTrainingData(self):
        accelerometer = np.load(self.trainingPath + "trainAccelerometer.npy")
        gravity = np.load(self.trainingPath + "trainGravity.npy")
        gyroscope = np.load(self.trainingPath + "trainGyroscope.npy")
        linearAcceleration = np.load(self.trainingPath + "trainLinearAcceleration.npy")
        magnetometer = np.load(self.trainingPath + "trainMagnetometer.npy")
        ##labels = np.load(self.trainingPath + "trainLabels.npy")
        return accelerometer, gravity, gyroscope, linearAcceleration, magnetometer

    def loadOriginalTestData(self):
        accelerometer = np.load(self.testPath + "testAccelerometer.npy")
        gravity = np.load(self.testPath + "testGravity.npy")
        gyroscope = np.load(self.testPath + "testGyroscope.npy")
        linearAcceleration = np.load(self.testPath + "testLinearAcceleration.npy")
        magnetometer = np.load(self.testPath + "testMagnetometer.npy")
        ##labels = np.load(self.testPath + "testLabels.npy")
        return accelerometer, gravity, gyroscope, linearAcceleration, magnetometer

    def loadTrainingLabels(self):
        return np.load(self.trainingPath + "trainLabels.npy")

    def loadTestLabels(self):
        return np.load(self.testPath + "testLabels.npy")

    def loadData(self, savePath, saveName):
        data = np.load(self.savePath + savePath + "\\" + saveName + ".npy")
        return data


    def saveData(self, savePath,saveName, data):
        np.save(self.savePath + savePath + "\\" + saveName, data)

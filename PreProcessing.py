##PreProcessing

from scipy import signal
from sklearn import preprocessing

class PreProcessing:

    def __init__(self):
        self.data = []
        self.sampleRate = 800

    def adaptSampleRate(self, accelerometer, gravity, gyroscope, linearAcceleration, magnetometer):
        if len(accelerometer[1, :, 1]) != self.sampleRate:
            accelerometer = signal.resample(x=accelerometer[:, :, :], num=800, axis=1)
        if len(gravity[1, :, 1]) != self.sampleRate:
            gravity = signal.resample(x=gravity[:, :, :], num=800, axis=1)
        if len(gyroscope[1, :, 1]) != self.sampleRate:
            gyroscope = signal.resample(x=gyroscope[:, :, :], num=800, axis=1)
        if len(linearAcceleration[1, :, 1]) != self.sampleRate:
            linearAcceleration = signal.resample(x=linearAcceleration[:, :, :], num=800, axis=1)
        if len(magnetometer[1, :, 1]) != self.sampleRate:
            magnetometer = signal.resample(x=magnetometer[:, :, :], num=800, axis=1)
        return accelerometer, gravity, gyroscope, linearAcceleration, magnetometer


    def normalizeAndScal(self, accelerometer, gravity, gyroscope, linearAcceleration, magnetometer):
        for i in range(1, 1692):
            accelerometer[i, :, :] = preprocessing.normalize(preprocessing.scale(accelerometer[i, :, :], axis=0), axis=0)
            gravity[i, :, :] = preprocessing.normalize(preprocessing.scale(gravity[i, :, :], axis=0), axis=0)
            gyroscope[i, :, :] = preprocessing.normalize(preprocessing.scale(gyroscope[i, :, :], axis=0), axis=0)
            linearAcceleration[i, :, :] = preprocessing.normalize(preprocessing.scale(linearAcceleration[i, :, :], axis=0), axis=0)
            magnetometer[i, :, :] = preprocessing.normalize(preprocessing.scale(magnetometer[i, :, :], axis=0), axis=0)
        return accelerometer, gravity, gyroscope, linearAcceleration, magnetometer

    def doAllPreProcessing(self, accelerometer, gravity, gyroscope, linearAcceleration, magnetometer):
        accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = self.adaptSampleRate(accelerometer, gravity, gyroscope, linearAcceleration, magnetometer)
        return self.normalizeAndScal(accelerometer, gravity, gyroscope, linearAcceleration, magnetometer)

    def simplePreProcessing(self, accelerometer, gravity, gyroscope, linearAcceleration, magnetometer):
        accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = self.adaptSampleRate(accelerometer,
                                                                                                   gravity, gyroscope,
                                                                                                   linearAcceleration,
                                                                                                   magnetometer)
        accelerometer = accelerometer - accelerometer.mean()
        gravity = gravity - gravity.mean()
        gyroscope = gyroscope - gyroscope.mean()
        linearAcceleration = linearAcceleration - linearAcceleration.mean()
        magnetometer = magnetometer - magnetometer.mean()

        accelerometer = accelerometer / accelerometer.std()
        gravity = gravity / gravity.std()
        gyroscope = gyroscope / gyroscope.std()
        linearAcceleration = linearAcceleration / linearAcceleration.std()
        magnetometer = magnetometer / magnetometer.std()

        return accelerometer, gravity, gyroscope, linearAcceleration, magnetometer

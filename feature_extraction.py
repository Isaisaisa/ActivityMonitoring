import os

import pandas as pd
import sklearn.preprocessing
import tsfresh
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config

accelerometer = config.TRAININGFILES + '\\trainAccelerometer.npy'
accelerometer_data = np.load(accelerometer)
plot = False


def normalize_data(timesequence):
    # shape is (1692, 800, 3)
    scale = sklearn.preprocessing.scale(timesequence[:, :], axis=0)
    # shape is (800, 3)
    norm = sklearn.preprocessing.normalize(scale[:, :], axis=0)
    if plot:
        t = np.arange(0., 4000., 4000 / timesequence.shape[0])
        plt.plot(t, norm[:, 0], 'b--', t, norm[:, 1], 'g--', t, norm[:, 2], 'r--')
        plt.show()
    return norm


def feature_extraction_process():
    # Normalzation of data
    if not os.path.exists('norm_accelerometer.npy'):
        print('file does not exist')
        normalized_data = []
        for i in range(accelerometer_data.shape[0]):
            normal = normalize_data(accelerometer_data[i])
            normalized_data.append(normal)
        np.save('norm_accelerometer.npy', normalized_data)
    else:
        normalized_data = np.load('norm_accelerometer.npy')
    # Feature extraction
    columns = list(range(normalized_data.shape[0]))
    a = normalized_data[:, :, 0]
    print(a.shape)
    print(columns)
    df = pd.DataFrame(np.transpose(normalized_data[:,:,0]), columns=columns)
    print(df.shape)
    tsfresh.extract_features(df, column_id='id')




# tsfresh.extract_features()
feature_extraction_process()
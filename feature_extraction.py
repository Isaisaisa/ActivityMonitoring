import os

import pandas as pd
import sklearn.preprocessing
import tsfresh
from tsfresh.feature_extraction import feature_calculators
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config
from scipy.signal import find_peaks, find_peaks_cwt

accelerometer = config.TRAININGFILES + '\\trainAccelerometer.npy'
accelerometer_data = np.load(accelerometer)
plot = False


def normalize_data(timesequence):
    # shape is (1692, 800, 3) num rows = 1692, columns = 800
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
    # 1692
    # columns = ['id'] + list(range(normalized_data.shape[0]))
    # # 800
    # index = range(normalized_data.shape[1])
    # my_data = normalized_data[:,:,0].T
    # print(my_data.shape)
    # new_col = np.array(range(normalized_data.shape[1]))[..., None]  # None keeps (n, 1) shape
    # all_data = np.append(new_col, my_data, axis=1)
    # # df.shape => (1692, 801) (rows, columns)
    # df = pd.DataFrame(all_data, index=index, columns=columns)
    # df2 = tsfresh.extract_features(df, column_id='id')
    # Does not really make sense, because we normalized timeseries so it is close to 0
    mean = np.mean(normalized_data, axis=1)
    median = np.median(normalized_data, axis=1)
    maxi = np.max(normalized_data, axis=1)
    mini = np.min(normalized_data, axis=1)
    # peaks_arr = np.empty((1692, 800, 3))
    peaks_arr = []
    # only of first timeseries of three
    for idx, timeserie in enumerate(normalized_data[:,:,0]):
        # print(timeserie.shape)
        peaks = find_peaks(timeserie, distance=20)
        peaks_arr.append(list(peaks[0]))
    ## np.arange(0., 4000., step=5)
    t = np.arange(start=0., stop=4000., step=4000 / normalized_data[0,:,0].shape[0])
    number_sequence = 200
    print(np.array(peaks_arr[0])*5)
    plt.plot(t, normalized_data[number_sequence,:,0], 'b-',
             np.array(peaks_arr[number_sequence])*5, normalized_data[number_sequence,:,0][peaks_arr[number_sequence]],'r+')
    plt.show()

feature_extraction_process()
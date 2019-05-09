import os

import pandas as pd
import sklearn.preprocessing
import tsfresh
from tsfresh.feature_extraction import feature_calculators
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import config_dev as config
from scipy.signal import find_peaks, find_peaks_cwt

accelerometer = config.TRAININGFILES + '\\trainAccelerometer.npy'
accelerometer_data = np.load(accelerometer)
plot = True


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
    # Does not really make sense, because we normalized timeseries so it is close to 0
    mean = np.mean(normalized_data, axis=1)
    median = np.median(normalized_data, axis=1)
    maxi = np.max(normalized_data, axis=1)
    mini = np.min(normalized_data, axis=1)
    # peaks_arr = np.empty((1692, 800, 3))
    peaks_arr_first_channel = []
    peaks_arr_second_channel = []
    peaks_arr_third_channel = []
    # only of first timeseries of three
    for timeserie in normalized_data[:,:,0]:
        # prominence – Required prominence of peaks. Either a number, ``None``, an array matching `x` or a 2-element
        # sequence of the former. The first element is always interpreted as the minimal and the second, if supplied,
        # as the maximal required prominence.
        peaks = find_peaks(timeserie, prominence=0.009)
        peaks_arr_first_channel.append(list(peaks[0]))
    ## np.arange(0., 4000., step
    if plot:
        t = np.arange(start=0., stop=4000., step=4000 / normalized_data[0,:,0].shape[0])
        number_sequences = [2, 10, 15, 200]
        print(np.array(peaks_arr_first_channel[0])*5)
        for num in number_sequences:
            plt.plot(t, normalized_data[num,:,0], 'b-',
                     np.array(peaks_arr_first_channel[num])*5, normalized_data[num,:,0][peaks_arr_first_channel[num]],'r+')
            plt.show()

    for timeserie in normalized_data[:, :, 1]:
        peaks = find_peaks(timeserie, prominence=0.009)
        peaks_arr_second_channel.append(list(peaks[0]))

    for timeserie in normalized_data[:, :, 2]:
        peaks = find_peaks(timeserie, prominence=0.009)
        peaks_arr_third_channel.append(list(peaks[0]))

feature_extraction_process()
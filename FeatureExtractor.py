## This class provides tools to extract features from the sensor data based on the tsfresh lib
## https://tsfresh.readthedocs.io/en/latest/index.html

import tsfresh as ts
import tsfresh.feature_extraction.feature_calculators as tsCalc
import numpy as np
from scipy.signal import find_peaks

class FeatureExtractor:

    def __init__(self):
        self.data = []

    ## Extract features from the data (3d data --> [N,T,S]), Calculate a feature vector for every channel (3 channels per sensor)
    ## 1692 executions, 3 channels --> 5076 feature vectors with xxx dimensions
    def extractFeatures(self,data):
        iVar = 30
        featureVectors = np.zeros(shape=(data.shape[0],90))
        # data.shape[0] -> 1692 total number of executions activities
        for i in range(0,data.shape[0]):
            # number of channels
            for j in range(0, 3):
                ## Calculate features
                featureVectors[i, j * iVar + 0] = tsCalc.mean(data[i, :, j])
                featureVectors[i, j * iVar + 1] = tsCalc.standard_deviation(data[i, :, j])
                featureVectors[i, j * iVar + 2] = tsCalc.abs_energy(data[i, :, j])
                featureVectors[i, j * iVar + 3] = tsCalc.maximum(data[i, :, j])
                featureVectors[i, j * iVar + 4] = tsCalc.minimum(data[i, :, j])

                featureVectors[i, j * iVar + 5] = tsCalc.count_above_mean(data[i, :, j])
                featureVectors[i, j * iVar + 6] = tsCalc.count_below_mean(data[i, :, j])
                featureVectors[i, j * iVar + 7] = tsCalc.first_location_of_minimum(data[i, :, j])
                featureVectors[i, j * iVar + 8] = tsCalc.first_location_of_maximum(data[i, :, j])
                featureVectors[i, j * iVar + 9] = tsCalc.kurtosis(data[i, :, j])

                featureVectors[i, j * iVar + 10] = tsCalc.variance(data[i, :, j])
                featureVectors[i, j * iVar + 11] = tsCalc.skewness(data[i, :, j])
                featureVectors[i, j * iVar + 12] = tsCalc.sum_values(data[i, :, j])
                featureVectors[i, j * iVar + 13] = tsCalc.percentage_of_reoccurring_datapoints_to_all_datapoints(data[i, :, j])
                featureVectors[i, j * iVar + 14] = tsCalc.mean_second_derivative_central(data[i, :, j])

                data_f = data[i, :, j]
                A = np.fft.fft(data_f)  # fourier transformation
                ampl = np.abs(A)  # amplitude spectrum
                power = np.abs(A) ** 2  # power spectrum

                featureVectors[i, j * iVar + 15] = tsCalc.maximum(A)
                featureVectors[i, j * iVar + 16] = tsCalc.mean(A)
                featureVectors[i, j * iVar + 17] = tsCalc.sample_entropy(A)
                featureVectors[i, j * iVar + 18] = tsCalc.abs_energy(A)
                featureVectors[i, j * iVar + 19] = tsCalc.first_location_of_maximum(A)

                featureVectors[i, j * iVar + 20] = tsCalc.maximum(ampl)
                featureVectors[i, j * iVar + 21] = tsCalc.mean(ampl)
                featureVectors[i, j * iVar + 22] = tsCalc.sample_entropy(ampl)
                featureVectors[i, j * iVar + 23] = tsCalc.abs_energy(ampl)
                featureVectors[i, j * iVar + 24] = tsCalc.first_location_of_maximum(ampl)

                featureVectors[i, j * iVar + 25] = tsCalc.maximum(power)
                featureVectors[i, j * iVar + 26] = tsCalc.mean(power)
                featureVectors[i, j * iVar + 27] = tsCalc.sample_entropy(power)
                featureVectors[i, j * iVar + 28] = tsCalc.abs_energy(power)
                featureVectors[i, j * iVar + 29] = tsCalc.first_location_of_maximum(power)

                # # number of prominence peaks in sequence
                # featureVectors[i, j * 5 + 15] = len(find_peaks(data[i, :, j], prominence=0.009)[0])
                # # crossing the x-axis at 0.0
                # featureVectors[i, j * 5 + 16] = tsCalc.number_crossing_m(data[i, :, j], 0.0)

        return featureVectors

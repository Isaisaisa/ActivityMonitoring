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
        featureVectors = np.zeros(shape=(data.shape[0],45))
        for i in range(0,data.shape[0]):
            for j in range(0, 3):
                ## Calculate ten features
                featureVectors[i, j * 5 + 0] = tsCalc.mean(data[i, :, j])
                featureVectors[i, j * 5 + 1] = tsCalc.standard_deviation(data[i, :, j])
                featureVectors[i, j * 5 + 2] = tsCalc.abs_energy(data[i, :, j])
                featureVectors[i, j * 5 + 3] = tsCalc.maximum(data[i, :, j])
                featureVectors[i, j * 5 + 4] = tsCalc.minimum(data[i, :, j])

                featureVectors[i, j * 5 + 5] = tsCalc.count_above_mean(data[i, :, j])
                featureVectors[i, j * 5 + 6] = tsCalc.count_below_mean(data[i, :, j])
                featureVectors[i, j * 5 + 7] = tsCalc.first_location_of_minimum(data[i, :, j])
                featureVectors[i, j * 5 + 8] = tsCalc.first_location_of_maximum(data[i, :, j])
                featureVectors[i, j * 5 + 9] = tsCalc.kurtosis(data[i, :, j])

                featureVectors[i, j * 5 + 10] = tsCalc.variance(data[i, :, j])
                featureVectors[i, j * 5 + 11] = tsCalc.skewness(data[i, :, j])
                featureVectors[i, j * 5 + 12] = tsCalc.sum_values(data[i, :, j])
                featureVectors[i, j * 5 + 13] = tsCalc.percentage_of_reoccurring_datapoints_to_all_datapoints(data[i, :, j])
                featureVectors[i, j * 5 + 14] = tsCalc.mean_second_derivative_central(data[i, :, j])

                # number of prominence peaks in sequence
                featureVectors[i, j * 5 + 15] = len(find_peaks(data[i, :, j], prominence=0.009)[0])
        return featureVectors

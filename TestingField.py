import DataLoader as loader
import numpy as np
import PreProcessing as preProcessor
import FeatureExtractor as fEx
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as acc
import meanAveragePrecision as meanP
import MlpClassifier as mlpc


# Load the data and Preprocessing
LOAD_DATA_AND_PREPROCESS = True
# Save preprocessed data
SAVE_PREPROCESSED_DATA = True
# Load the saved preprocessed data
LOAD_PREPROCESSED_DATA = True
# Feature Extraction
FEATURE_EXTRACTION = True
# Save the extracted Features
SAVE_FEATURE_VECTORS = True
# Load saved feature vectors
LOAD_FEATURE_VECTORS = True
# Feature Selection
SELECT_FEATURES = False
# Train the MLP and use it to classify test data
TRAIN_AND_CLASSIFY = True
# Swap between two version of MLP
# 1. sklearn MLP --> False
# 2. Keras MLP --> True
USE_KERAS_MLP = True

mlp = None

# MLP settings for sklearn MLP (the settings fpr Keras MLP --> MLPClassifier.py)

parameter = {  # 0.25 and 0.11
    'solver' : 'adam',
    'activation'  : 'relu',
    'validation_fraction' : 0.15,
    'early_stopping' : True,
    'learning_rate_init' : 0.001,
    'alpha' : 1e-4,
    'tol' : 0.03,
    'n_iter_no_change' : 50,
    'learning_rate' : 'adaptive',
    'hidden_layer_sizes' : (450,450,450,450,450),
    'batch_size' : 250,
    'max_iter' : 2000,
    'shuffle' : True,
    'random_state' : True,
    'verbose' : True
}

# Load the data and preprocess it
dataLoader = loader.DataLoader()
# Do simple preProcessing
if(LOAD_DATA_AND_PREPROCESS):
    preProcess = preProcessor.PreProcessing()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = dataLoader.loadOriginalTrainingData()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = preProcess.simplePreProcessing(accelerometer,
                                                                                                        gravity,
                                                                                                        gyroscope,
                                                                                                        linearAcceleration,
                                                                                                        magnetometer)

# Save PreProcessed Data
if (SAVE_PREPROCESSED_DATA):
    dataLoader.saveData("PreProcessedData\\Training", "accelerometer", accelerometer)
    dataLoader.saveData("PreProcessedData\\Training", "gravity", gravity)
    dataLoader.saveData("PreProcessedData\\Training", "gyroscope", gyroscope)
    dataLoader.saveData("PreProcessedData\\Training", "linearAcceleration", linearAcceleration)
    dataLoader.saveData("PreProcessedData\\Training", "magnetometer", magnetometer)

# Load the preprocessed data
if (LOAD_PREPROCESSED_DATA):
    accelerometer = dataLoader.loadData("PreProcessedData\\Training", "accelerometer")
    gravity = dataLoader.loadData("PreProcessedData\\Training", "gravity")
    gyroscope = dataLoader.loadData("PreProcessedData\\Training", "gyroscope")
    linearAcceleration = dataLoader.loadData("PreProcessedData\\Training", "linearAcceleration")
    magnetometer = dataLoader.loadData("PreProcessedData\\Training", "magnetometer")


# Feature Extraction
if (FEATURE_EXTRACTION):
    fExtractor = fEx.FeatureExtractor()
    accelerometerFeatureVector = fExtractor.extractFeatures(accelerometer)
    gravityFeatureVector = fExtractor.extractFeatures(gravity)
    gyroscopeFeatureVector = fExtractor.extractFeatures(gyroscope)
    linearAccelerationFeatureVector = fExtractor.extractFeatures(linearAcceleration)
    magnetometerFeatureVector = fExtractor.extractFeatures(magnetometer)

# Save extracted Features
if(SAVE_FEATURE_VECTORS):
    dataLoader.saveData("Features\\Training", "accelerometer", accelerometerFeatureVector)
    dataLoader.saveData("Features\\Training", "gravity", gravityFeatureVector)
    dataLoader.saveData("Features\\Training", "gyroscope", gyroscopeFeatureVector)
    dataLoader.saveData("Features\\Training", "linearAcceleration", linearAccelerationFeatureVector)
    dataLoader.saveData("Features\\Training", "magnetometer", magnetometerFeatureVector)

# Load the extracted Features
if (LOAD_FEATURE_VECTORS):
    accelerometerFeatureVector = dataLoader.loadData("Features\\Training", "accelerometer")
    gravityFeatureVector = dataLoader.loadData("Features\\Training", "gravity")
    gyroscopeFeatureVector = dataLoader.loadData("Features\\Training", "gyroscope")
    linearAccelerationFeatureVector = dataLoader.loadData("Features\\Training", "linearAcceleration")
    magnetometerFeatureVector = dataLoader.loadData("Features\\Training", "magnetometer")

# Train the Classifier
if(TRAIN_AND_CLASSIFY):
    classifier = MLPClassifier(**parameter)
    ##Load Labels
    labels = dataLoader.loadTrainingLabels()

    ## Feature Selection
    if (SELECT_FEATURES):
        import FeatureSelector as fs

        selector = fs.FeatureSelector()
        data = selector.selectAllFeatures(np.concatenate(
            (accelerometerFeatureVector,
             gravityFeatureVector,
             gyroscopeFeatureVector,
             linearAccelerationFeatureVector,
             magnetometerFeatureVector), axis=1))

    else:
        ## Create appropiate matrix for MLP training  => n_samples x m_features = 1692 x 75
        data = np.zeros(shape=(1692, 225))
        for i in range(0, 1692):
            data[i, :] = np.concatenate(([accelerometerFeatureVector[i, :], gravityFeatureVector[i, :],
                                          gyroscopeFeatureVector[i, :], linearAccelerationFeatureVector[i, :],
                                          magnetometerFeatureVector[i, :]]), axis=0)

    if USE_KERAS_MLP:
        mlp = mlpc.MlpClassifier(inputShape=data.shape[1])
        mlp.train(data, labels)

        # Predict class labels and probabilities for the classes
        predictedLabels = mlp.predicted_labels(data)
        predictedProbs = mlp.predict(data)
        # Accuracy
        accuracy = mlp.eval(data, labels)
        # Mean Average Precision
        ar, flo = meanP.computeMeanAveragePrecision(labels=labels, softmaxEstimations=predictedProbs)
        # F1 Score
        fscore = 'Not defined'

    else:
        classifier.fit(data, labels)

        # Classify training data with one class
        predictedLabels = classifier.predict(data)
        # Predict probabilities for every class
        predictedProbs = classifier.predict_proba(data)
        predictedProbsCorrection = (np.append(np.append(predictedProbs[:, 0:8], np.zeros((1692, 1)), axis=1), predictedProbs[:, 8:53], axis=1))
        # Mean Average Precision
        ar, flo = meanP.computeMeanAveragePrecision(labels=labels, softmaxEstimations=predictedProbsCorrection)
        # Accuracy
        accuracy = acc.accuracy_score(labels, predictedLabels)
        # F1Score
        fscore = acc.f1_score(y_true=labels, y_pred=predictedLabels, average='micro')

    print('----Training completed----')
    print('Results on the training data')
    print('Accuracy: ', accuracy)
    print('F1-Score: ', fscore)
    print('Mean Average Precision:', ar)



# Classify Data -----------------------------------


# Do simple preProcessing
if (LOAD_DATA_AND_PREPROCESS):
    preProcess = preProcessor.PreProcessing()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = dataLoader.loadOriginalTestData()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = preProcess.simplePreProcessing(accelerometer,
                                                                                                         gravity,
                                                                                                         gyroscope,
                                                                                                         linearAcceleration,magnetometer)

# Save PreProcessed Data
##
if (SAVE_PREPROCESSED_DATA):
    dataLoader.saveData("PreProcessedData\\Testing", "accelerometer", accelerometer)
    dataLoader.saveData("PreProcessedData\\Testing", "gravity", gravity)
    dataLoader.saveData("PreProcessedData\\Testing", "gyroscope", gyroscope)
    dataLoader.saveData("PreProcessedData\\Testing", "linearAcceleration", linearAcceleration)
    dataLoader.saveData("PreProcessedData\\Testing", "magnetometer", magnetometer)

# Load the preprocessed data
if (LOAD_PREPROCESSED_DATA):
    accelerometer = dataLoader.loadData("PreProcessedData\\Testing", "accelerometer")
    gravity = dataLoader.loadData("PreProcessedData\\Testing", "gravity")
    gyroscope = dataLoader.loadData("PreProcessedData\\Testing", "gyroscope")
    linearAcceleration = dataLoader.loadData("PreProcessedData\\Testing", "linearAcceleration")
    magnetometer = dataLoader.loadData("PreProcessedData\\Testing", "magnetometer")


# Feature Extraction
if (FEATURE_EXTRACTION):
    fExtractor = fEx.FeatureExtractor()
    accelerometerFeatureVector = fExtractor.extractFeatures(accelerometer)
    gravityFeatureVector = fExtractor.extractFeatures(gravity)
    gyroscopeFeatureVector = fExtractor.extractFeatures(gyroscope)
    linearAccelerationFeatureVector = fExtractor.extractFeatures(linearAcceleration)
    magnetometerFeatureVector = fExtractor.extractFeatures(magnetometer)

# Save extracted Features
if(SAVE_FEATURE_VECTORS):
    dataLoader.saveData("Features\\Testing", "accelerometer", accelerometerFeatureVector)
    dataLoader.saveData("Features\\Testing", "gravity", gravityFeatureVector)
    dataLoader.saveData("Features\\Testing", "gyroscope", gyroscopeFeatureVector)
    dataLoader.saveData("Features\\Testing", "linearAcceleration", linearAccelerationFeatureVector)
    dataLoader.saveData("Features\\Testing", "magnetometer", magnetometerFeatureVector)

# Load the extracted Features
if (LOAD_FEATURE_VECTORS):
    accelerometerFeatureVector = dataLoader.loadData("Features\\Testing", "accelerometer")
    gravityFeatureVector = dataLoader.loadData("Features\\Testing", "gravity")
    gyroscopeFeatureVector = dataLoader.loadData("Features\\Testing", "gyroscope")
    linearAccelerationFeatureVector = dataLoader.loadData("Features\\Testing", "linearAcceleration")
    magnetometerFeatureVector = dataLoader.loadData("Features\\Testing", "magnetometer")

# Classify
if(TRAIN_AND_CLASSIFY):
    ##Load Labels
    labels = dataLoader.loadTestLabels()

    ## Feature Selection
    if (SELECT_FEATURES):
        import FeatureSelector as fs

        selector = fs.FeatureSelector()

        data = selector.selectAllFeatures(np.concatenate(
            (accelerometerFeatureVector,
             gravityFeatureVector,
             gyroscopeFeatureVector,
             linearAccelerationFeatureVector,
             magnetometerFeatureVector), axis=1))

    else:
        ## Create appropiate matrix for MLP training  => n_samples x m_features = 1692 x 75
        data = np.zeros(shape=(1705, 225))
        for i in range(0, 1692):
            data[i, :] = np.concatenate(([accelerometerFeatureVector[i, :], gravityFeatureVector[i, :],
                                          gyroscopeFeatureVector[i, :], linearAccelerationFeatureVector[i, :],
                                          magnetometerFeatureVector[i, :]]), axis=0)

    ##Classify test data
    if USE_KERAS_MLP:
        predictedLabels = mlp.predicted_labels(data)
        predictedProbs = mlp.predict(data)
        # Accuracy
        accuracy = mlp.eval(data, labels)
        # Mean Average Precision
        ar, flo = meanP.computeMeanAveragePrecision(labels=labels, softmaxEstimations=predictedProbs)
        # F1 Score
        fscore = 'Not defined'
    else:
        predictedLabels = classifier.predict(data)
        # Predict Probabilites
        predictedProbs = classifier.predict_proba(data)
        # In the data is no sample for class 9, add a probability for this with 0
        predictedProbsCorrection = (np.append(np.append(predictedProbs[:, 0:8], np.zeros((1705, 1)), axis=1), predictedProbs[:, 8:53], axis=1))
        # Mean Average Precision
        ar, flo = meanP.computeMeanAveragePrecision(labels=labels, softmaxEstimations= predictedProbsCorrection )
        # Accuracy
        accuracy = acc.accuracy_score(labels, predictedLabels)
        #F1 Score
        fscore = acc.f1_score(y_true = labels,y_pred = predictedLabels, average='micro')

    print('---------')
    print('Results on the test data')
    print('Accuracy: ', accuracy)
    print('F1-Score: ', fscore)
    print('Mean Average Precision(with correction):', ar)

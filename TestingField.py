import DataLoader as loader
import numpy as np
import PreProcessing as preProcessor
import FeatureExtractor as fEx
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as acc
import MlpClassifier as mlpc

SELECT_FEATURES_ALL = False
SELECT_FEATURES_PER_SENSOR = False

### Training
SIMPLE_PREPROCESSING = False
SAVE_PREPROCESSING = False
LOAD_PREPROCESSING = False
FEATURE_EXTRACTION = False
LOAD_FEATURES = True
TRAIN_CLASSIFIER = True

### Evaluation
TEST_DATA_SIMPLE_PREPROCESSING = False
TEST_DATA_SAVE_PREPROCESSING = False
TEST_DATA_LOAD_PREPROCESSING = False
TEST_DATA_FEATURE_EXTRACTION = False
TEST_DATA_LOAD_FEATURES = True
TEST_CLASSIFIER = True

mlp = None
## Load the data and preprocess it
dataLoader = loader.DataLoader()

## Do simple preProcessing
if(SIMPLE_PREPROCESSING):
    preProcess = preProcessor.PreProcessing()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = dataLoader.loadOriginalTrainingData()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = preProcess.simplePreProcessing(accelerometer,
                                                                                                        gravity,
                                                                                                        gyroscope,
                                                                                                        linearAcceleration,
                                                                                                        magnetometer)


# Save PreProcessed Data
if (SAVE_PREPROCESSING):
    dataLoader.saveData("PreProcessedData\\Training", "accelerometer", accelerometer)
    dataLoader.saveData("PreProcessedData\\Training", "gravity", gravity)
    dataLoader.saveData("PreProcessedData\\Training", "gyroscope", gyroscope)
    dataLoader.saveData("PreProcessedData\\Training", "linearAcceleration", linearAcceleration)
    dataLoader.saveData("PreProcessedData\\Training", "magnetometer", magnetometer)



## Load the preprocessed data
if (LOAD_PREPROCESSING):
    accelerometer = dataLoader.loadData("PreProcessedData\\Training", "accelerometer")
    gravity = dataLoader.loadData("PreProcessedData\\Training", "gravity")
    gyroscope = dataLoader.loadData("PreProcessedData\\Training", "gyroscope")
    linearAcceleration = dataLoader.loadData("PreProcessedData\\Training", "linearAcceleration")
    magnetometer = dataLoader.loadData("PreProcessedData\\Training", "magnetometer")


## Feature Extraction and Save Features Vectors
if (FEATURE_EXTRACTION):
    fExtractor = fEx.FeatureExtractor()
    accelerometerFeatureVector = fExtractor.extractFeatures(accelerometer)
    gravityFeatureVector = fExtractor.extractFeatures(gravity)
    gyroscopeFeatureVector = fExtractor.extractFeatures(gyroscope)
    linearAccelerationFeatureVector = fExtractor.extractFeatures(linearAcceleration)
    magnetometerFeatureVector = fExtractor.extractFeatures(magnetometer)


    ## Save extracted Features
    dataLoader.saveData("Features\\Training", "accelerometer", accelerometerFeatureVector)
    dataLoader.saveData("Features\\Training", "gravity", gravityFeatureVector)
    dataLoader.saveData("Features\\Training", "gyroscope", gyroscopeFeatureVector)
    dataLoader.saveData("Features\\Training", "linearAcceleration", linearAccelerationFeatureVector)
    dataLoader.saveData("Features\\Training", "magnetometer", magnetometerFeatureVector)

## Load the extracted Features
if (LOAD_FEATURES):
    accelerometerFeatureVector = dataLoader.loadData("Features\\Training", "accelerometer")
    gravityFeatureVector = dataLoader.loadData("Features\\Training", "gravity")
    gyroscopeFeatureVector = dataLoader.loadData("Features\\Training", "gyroscope")
    linearAccelerationFeatureVector = dataLoader.loadData("Features\\Training", "linearAcceleration")
    magnetometerFeatureVector = dataLoader.loadData("Features\\Training", "magnetometer")




## Train the Classifier
if(TRAIN_CLASSIFIER):
    classifier = MLPClassifier(solver='adam', learning_rate='adaptive', hidden_layer_sizes=(450,450),tol=0.0000100, batch_size = 1000,  max_iter=2000, shuffle=True, random_state=True, verbose= 10)
    ##Load Labels
    labels = dataLoader.loadTrainingLabels()
    data = None
    ## Feature Selection
    if (SELECT_FEATURES_ALL):
        import FeatureSelector as fs

        selector = fs.FeatureSelector()

        data = selector.selectAllFeatures(np.concatenate(
            (accelerometerFeatureVector,
             gravityFeatureVector,
             gyroscopeFeatureVector,
             linearAccelerationFeatureVector,
             magnetometerFeatureVector), axis=1))

    if (SELECT_FEATURES_PER_SENSOR):
        accelerometerFeatureVector = selector.selectFeatures(accelerometerFeatureVector)
        gravityFeatureVector = selector.selectFeatures(gravityFeatureVector)
        gyroscopeFeatureVector = selector.selectFeatures(gyroscopeFeatureVector)
        linearAccelerationFeatureVector = selector.selectFeatures(linearAccelerationFeatureVector)
        magnetometerFeatureVector = selector.selectFeatures(magnetometerFeatureVector)

        ## Create appropiate matrix for MLP training  => n_samples x m_features = 1692 x 75
        data = np.zeros(shape = (1692,50))
        for i in range(0, 1692):
            data[i, :] = np.concatenate(([accelerometerFeatureVector[i, :], gravityFeatureVector[i, :], gyroscopeFeatureVector[i, :], linearAccelerationFeatureVector[i, :], magnetometerFeatureVector[i, :]]), axis = 0)




    if (~SELECT_FEATURES_ALL & ~SELECT_FEATURES_PER_SENSOR):
        ## Create appropiate matrix for MLP training  => n_samples x m_features = 1692 x 75
        data = np.zeros(shape=(1692, 225))
        for i in range(0, 1692):
            data[i, :] = np.concatenate(([accelerometerFeatureVector[i, :], gravityFeatureVector[i, :],
                                          gyroscopeFeatureVector[i, :], linearAccelerationFeatureVector[i, :],
                                          magnetometerFeatureVector[i, :]]), axis=0)
    #classifier.fit(data, labels)
    mlp = mlpc.MlpClassifier()
    mlp.train(data, labels)



## Classify Data -----------------------------------


## Do simple preProcessing
if (TEST_DATA_SIMPLE_PREPROCESSING):
    preProcess = preProcessor.PreProcessing()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = dataLoader.loadOriginalTestData()
    accelerometer, gravity, gyroscope, linearAcceleration, magnetometer = preProcess.simplePreProcessing(accelerometer,
                                                                                                         gravity,
                                                                                                         gyroscope,
                                                                                                         linearAcceleration,magnetometer)

# Save PreProcessed Data
##
if (TEST_DATA_SAVE_PREPROCESSING):
    dataLoader.saveData("PreProcessedData\\Testing", "accelerometer", accelerometer)
    dataLoader.saveData("PreProcessedData\\Testing", "gravity", gravity)
    dataLoader.saveData("PreProcessedData\\Testing", "gyroscope", gyroscope)
    dataLoader.saveData("PreProcessedData\\Testing", "linearAcceleration", linearAcceleration)
    dataLoader.saveData("PreProcessedData\\Testing", "magnetometer", magnetometer)

## Load the preprocessed data
if (TEST_DATA_LOAD_PREPROCESSING):
    accelerometer = dataLoader.loadData("PreProcessedData\\Testing", "accelerometer")
    gravity = dataLoader.loadData("PreProcessedData\\Testing", "gravity")
    gyroscope = dataLoader.loadData("PreProcessedData\\Testing", "gyroscope")
    linearAcceleration = dataLoader.loadData("PreProcessedData\\Testing", "linearAcceleration")
    magnetometer = dataLoader.loadData("PreProcessedData\\Testing", "magnetometer")


## Feature Extraction and Save Features Vectors
if (TEST_DATA_FEATURE_EXTRACTION):
    fExtractor = fEx.FeatureExtractor()
    accelerometerFeatureVector = fExtractor.extractFeatures(accelerometer)
    gravityFeatureVector = fExtractor.extractFeatures(gravity)
    gyroscopeFeatureVector = fExtractor.extractFeatures(gyroscope)
    linearAccelerationFeatureVector = fExtractor.extractFeatures(linearAcceleration)
    magnetometerFeatureVector = fExtractor.extractFeatures(magnetometer)


    ## Save extracted Features
    dataLoader.saveData("Features\\Testing", "accelerometer", accelerometerFeatureVector)
    dataLoader.saveData("Features\\Testing", "gravity", gravityFeatureVector)
    dataLoader.saveData("Features\\Testing", "gyroscope", gyroscopeFeatureVector)
    dataLoader.saveData("Features\\Testing", "linearAcceleration", linearAccelerationFeatureVector)
    dataLoader.saveData("Features\\Testing", "magnetometer", magnetometerFeatureVector)

## Load the extracted Features
if (TEST_DATA_LOAD_FEATURES):
    accelerometerFeatureVector = dataLoader.loadData("Features\\Testing", "accelerometer")
    gravityFeatureVector = dataLoader.loadData("Features\\Testing", "gravity")
    gyroscopeFeatureVector = dataLoader.loadData("Features\\Testing", "gyroscope")
    linearAccelerationFeatureVector = dataLoader.loadData("Features\\Testing", "linearAcceleration")
    magnetometerFeatureVector = dataLoader.loadData("Features\\Testing", "magnetometer")






if(TEST_CLASSIFIER):
    ##Load Labels
    labels = dataLoader.loadTestLabels()

    ## Feature Selection
    if (SELECT_FEATURES_ALL):
        import FeatureSelector as fs

        selector = fs.FeatureSelector()

        data = selector.selectAllFeatures(np.concatenate(
            (accelerometerFeatureVector,
             gravityFeatureVector,
             gyroscopeFeatureVector,
             linearAccelerationFeatureVector,
             magnetometerFeatureVector), axis=1))

    if (SELECT_FEATURES_PER_SENSOR):
        accelerometerFeatureVector = selector.selectFeatures(accelerometerFeatureVector)
        gravityFeatureVector = selector.selectFeatures(gravityFeatureVector)
        gyroscopeFeatureVector = selector.selectFeatures(gyroscopeFeatureVector)
        linearAccelerationFeatureVector = selector.selectFeatures(linearAccelerationFeatureVector)
        magnetometerFeatureVector = selector.selectFeatures(magnetometerFeatureVector)

        ## Create appropiate matrix for MLP training  => n_samples x m_features = 1692 x 75
        data = np.zeros(shape = (1705,50))
        for i in range(0, 1705):
            data[i, :] = np.concatenate(([accelerometerFeatureVector[i, :], gravityFeatureVector[i, :], gyroscopeFeatureVector[i, :], linearAccelerationFeatureVector[i, :], magnetometerFeatureVector[i, :]]), axis = 0)


    if (~SELECT_FEATURES_ALL & ~SELECT_FEATURES_PER_SENSOR):
        ## Create appropiate matrix for MLP training  => n_samples x m_features = 1692 x 75
        data = np.zeros(shape=(1705, 225))
        for i in range(0, 1705):
            data[i, :] = np.concatenate(([accelerometerFeatureVector[i, :], gravityFeatureVector[i, :],
                                          gyroscopeFeatureVector[i, :], linearAccelerationFeatureVector[i, :],
                                          magnetometerFeatureVector[i, :]]), axis=0)
    #predictedLabels = classifier.predict(data)
    accuracy = mlp.eval(data, labels)
    predictedLabels = mlp.predict(data)




#accuracy = acc.accuracy_score(labels, predictedLabels)
#fscore = acc.f1_score(y_true = labels,y_pred = predictedLabels, average='micro')
print(accuracy)
#print(fscore)


# ActivityMonitoring

### Installing libraries
    
    pip install -r requirements.txt
    
### Configure config file

We have a config file in this project calling ```` config.py ````. This file should be copied and the copied file should 
be renamed to ``` config_dev.py ```. Then you can configure the paths to the origin data for training and testing.
The constant ``` SAVEPATH ``` defines where the processed data will be saved e.g. preprocessed data or the extracted features.
The respective folders are created when starting the process/program.
    
### Run project

    python PATH\TO\PROJECT\ActivityMonitoring\TestingField.py
    
### Project structure
For each Process step we have a separated class/file.
The file ``` TestingField.py ``` starts all the Process at once and includes some boolean variables as switches.
    
    # Load the data and Preprocessing 
    LOAD_DATA_AND_PREPROCESS = False
    # Save preprocessed data
    SAVE_PREPROCESSED_DATA = False
    # Load the saved preprocessed data
    LOAD_PREPROCESSED_DATA = False
    # Feature Extraction
    FEATURE_EXTRACTION = False
    # Save the extracted Features
    SAVE_FEATURE_VECTORS = False
    # Load saved feature vectors
    LOAD_FEATURE_VECTORS = True
    # Feature Selection
    SELECT_FEATURES = True
    # Train the MLP and use it to classify test data
    TRAIN_AND_CLASSIFY = True
    # Swap between two version of MLP
    USE_KERAS_MLP = True
    

Set all switches to True when you run the program for the first time.
Only the last switch ``` USE_KERAS_MLP ``` can be set to True or False. It depends on which MLP you want to train.
If you use True then the Keras MLP will be used, if you use False then the Sklearn MLP will be trained.

When you start it for the first time, some new folders will be created under the path you specified under
```SAVEPATHH``` in the configuration file (see the heading [Configure config file](#Marker Header Configuration Configuration File)).

If you have already run the program and all files with the features etc. are created, then you can set most of the 
switches to ```False```, except ```LOAD_FEATURE_VECTORS, TRAIN_AND_CLASSIFY and maybe USE_KERAS_MLP```.


Further classes/files:

```DataLoader.py```: Loads the origin data for processing

```PreProcessing.py```: Prepossesses the data so it is normalized and the sequences have the same length

```FeatureExtractor.py```: Extracts handcrafted features chosen from the library tsfresh

```FeatureSelector.py```: Choose features using PCA 

```MlpClassifier.py```: Classifies the features using keras Multi Layer Perceptron (MLP) 

All process-specific settings can be read in the report (MLP settings or handcrafted features). 
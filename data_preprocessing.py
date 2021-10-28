import numpy as np

def data_preprocessing(x_train, x_test):
    """data preprocessing function; we return cleaned and processed data to go on with the analysis."""

    #1st -> remove missing features and select good ones:
    threshold=0.25
    ids_missing = discard_missing_features(x_train, threshold)
    x_train=select_good_features(x_train, ids_missing)
    x_test=select_good_features(x_test, ids_missing)
    
    #secondly -> let's standardize the data
    x_train=standardization(x_train)
    x_test=standardization(x_test)
    
    return x_train, x_test
    
def select_good_features(data_x, ids):
    """ return data thanks to the ids_missing list. """
    return data_x[:, ids]

def discard_missing_features(data_x, threshold):
    """return one boolean per column : true if the column is good ie has less undefined values than a given percentage"""
    percentage = np.mean(data_x == -999, axis=0)
    return (percentage < threshold)

def standardization(data_x):
    """ return standardized data:"""
    mean = np.mean(data_x, axis=0)
    std = np.std(data_x, axis=0)
    return (data_x-mean)/std
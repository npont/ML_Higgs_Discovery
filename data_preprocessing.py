import numpy as np

def data_preprocessing(x_train, x_test):
    """ data preprocessing function pipeline; returns the cleaned and processed data for analysis."""
    
    #first remove data which has below threshold level of data
    threshold = 0.25
    missing_ids = discard_missing_features(x_train, threshold) #find indices of bad data
    x_train = select_good_features(x_train, missing_ids)
    x_test = select_good_features(x_test, missing_ids)

    #x_train = replace_by_median(x_train)
    #x_test = replace_by_median(x_test)
            
    cat_feat_id = 18
    x_train, x_test = dummy_variables(x_train,x_test,cat_feat_id)
    
    x_train = standardization(x_train)
    x_test = standardization(x_test)

    return x_train, x_test


def dummy_variables(x_train, x_test, id_):
    """Function that converts the feature column represented by 
    its index (containing categorical data) to new columns as dummy variables."""
    min_ = 0 # should be 0
    max_ = 3 #should be 3
    
    #creates the zero columns
    dummy_var_train = np.zeros((x_train.shape[0],4))
    dummy_var_test = np.zeros((x_test.shape[0],4))
    
    for i in range(4):
        dummy_var_train[np.where(x_train[:,id_]==i),i] = 1 #replace by one based on values
        dummy_var_test[np.where(x_test[:,id_]==i),i] = 1
     
    x_train= np.delete(x_train,id_,axis=1) #delete the categorical variable
    x_test= np.delete(x_test,id_,axis=1)
    x_train = np.concatenate((x_train,dummy_var_train), axis=1)
    x_test = np.concatenate((x_test,dummy_var_test), axis=1)  
    
    return x_train, x_test

def select_good_features(data_x, ids):
    """ return data thanks to the ids_missing list. """
    return data_x[:, ids]

def replace_by_median(x, to_replace = -999):
    """ replaces bad values (-999) of column id_ by the median.  """
    for j in range(x.shape[1]):
        x[:,j] = np.where(x[:,j] == to_replace, np.median(x[:,j]!=to_replace), x[:,j])
    return x


def discard_missing_features(data_x, threshold):
    """return one boolean per column : true if the column is good ie has less undefined values than a given percentage"""
    percentage = np.mean(data_x == -999, axis=0)
    return (percentage < threshold)

def standardization(data_x):
    """ return standardized data:"""
    mean = np.mean(data_x, axis=0)
    std = np.std(data_x, axis=0)
    return (data_x-mean)/std
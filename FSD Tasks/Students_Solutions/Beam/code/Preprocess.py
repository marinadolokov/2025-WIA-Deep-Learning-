import numpy as np

def column_standardize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    
    df[new_column_name] = df[column_name].apply(standardize_columns)
    
    return df

def standardize_columns(arr):
    means = np.mean(arr, axis=0, keepdims=True)  
    stds = np.std(arr, axis=0, keepdims=True)    
    stds[stds == 0] = 1

    return (arr - means) / stds

#######################################################################################

def row_standardize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    
    df[new_column_name] = df[column_name].apply(standardize_rows)
    
    return df


def standardize_rows(arr):
    means = np.mean(arr, axis=1, keepdims=True)
    stds = np.std(arr, axis=1, keepdims=True)   
    
    stds[stds == 0] = 1

    return (arr - means) / stds

###########################################################################################

def array_standardize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    
    df[new_column_name] = df[column_name].apply(standardize_array)
    
    return df


def standardize_array(arr):
    means = np.mean(arr, keepdims=True)  
    stds = np.std(arr, keepdims=True) 
    stds[stds == 0] = 1
    
    return (arr - means) / stds

###############################################################################################

def column_normalize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    
    df[new_column_name] = df[column_name].apply(normalize_columns)
    
    return df


def normalize_columns(arr):
    mins = np.min(arr, axis=0, keepdims=True) 
    maxs = np.max(arr, axis=0, keepdims=True) 
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    return (arr - mins) / ranges

##############################################################################################

def row_normalize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    
    df[new_column_name] = df[column_name].apply(normalize_rows)
    
    return df


def normalize_rows(arr):
    mins = np.min(arr, axis=1, keepdims=True) 
    maxs = np.max(arr, axis=1, keepdims=True) 
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    return (arr - mins) / ranges


#################################################################################################

def array_normalize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    
    df[new_column_name] = df[column_name].apply(normalize_array)
    
    return df

def normalize_array(arr):
    mins = np.min(arr, keepdims=True)  
    maxs = np.max(arr, keepdims=True) 
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    return (arr - mins) / ranges

################################################################################################


def global_standardize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    all_data = np.vstack(df[column_name].values)
    mean = np.mean(all_data)
    std = np.std(all_data)
    
    if std == 0:
        df[new_column_name] = df[column_name].apply(lambda x: np.zeros_like(x))
        return df
    standardize = lambda arr: (arr - mean) / std
    df[new_column_name] = df[column_name].apply(standardize)
    
    return df

####################################################################

def global_min_max_normalize(df, column_name='sensor_data', new_column_name='sensor_data_prepro'):
    all_data = np.vstack(df[column_name].values)
    min = np.min(all_data)
    max = np.max(all_data)
    
    if max == min:
        df[new_column_name] = df[column_name].apply(lambda x: np.zeros_like(x))
        return df
    normalize = lambda arr: (arr - min) / (max - min)
    df[new_column_name] = df[column_name].apply(normalize)
    
    return df
    
# df = Funktion(df, 'sensor_data', 'sensor_data_prepro')

#Der Quelltext erstallt eine neue Spalte im Dataframe um die alten Daten nicht zu überschreiben
#Falls ihr links das _standardized löscht sollte es inplace funktionieren und ihr müsst euren Quelltext nicht anpassen

PREPROCESS_METHODS = [lambda x:x, column_standardize, row_standardize, array_standardize, column_normalize, row_normalize,
                      array_normalize, global_standardize, global_min_max_normalize]
import numpy as np
import pandas as pd
import pickle
import math

#APPROACH 1: Data augmentation by adding gaussian noise
#my is the mean, sigma indicates the standard deviation
#for example: my = 0, sigma = 1
def NoiseAdding(arr):

    noise = np.random.normal(0, 1, size = (arr.shape[0], arr.shape[1]))
    arr_with_noise = np.add(noise, arr)
    
    return arr_with_noise

#APPROACH 2: Data Augmentation by Scaling the amplitude by a factor, you might want to let the factor be a random value between two values
def AmplitudeScaling(arr):

    scaled_arr = np.multiply(arr, 1.1)

    return scaled_arr


#APPROACH 3: ZoomIn, start between 0 and 64, start indicates the starting point of the zoom-in
#ZoomIn takes half of the data and creates synthetic mid-points between two successive points 
#start = 32 creates a centralized ZoomIn
def ZoomIn(arr):

    zoomed_arr = np.zeros_like(arr)
    for k in range(32, 32+64):
        zoomed_arr[2*(k-32),:] = arr[k,:]
    for k in range(1,126,2):
        zoomed_arr[k,:] = (zoomed_arr[k-1,:] + zoomed_arr[k+1,:])/2
    zoomed_arr[127,:]=zoomed_arr[126,:]
    return zoomed_arr

#APPROACH 4: Flip the Array, an out of the air-approach, 
#the last point in time will be the first and vice versa
def Flip(arr):
    flip_arr = np.zeros_like(arr)
    for k in range(arr.shape[0]-1,-1,-1):
        flip_arr = arr[::-1]

    return flip_arr

#############################################################################################################################


#get the labels and data out of the given dataframe
#df is the dataframe we get out of test.pickle 
# with open(...) as file:
#     df = pickle.load(file)
# train_labels = df.label.to_numpy()
# train_data = np.stack(df.sensor_data.to_numpy())
# df_np = df.to_numpy()

#investigate how many samples of each label are in the dataset
#count = np.zeros(shape = (12))
#print(df_np.shape)
#for i in range (4053):
#    count[df_np[i,1]] += 1
#print(count)

#we calculate how many values we need to add per label to have the same amount as in label 0
#therefore we calculate how many "full" iterations and how many extra additional values we need to get to the aim of 1798 values from label 0
#for label 1: 1798 - 273 = 1525, floor(1525/273) = 5, 1525 - 5*273 = 160
#for label 2: 1798 - 227 = 1571, floor(1571/227) = 6, 1571 - 6*227 = 209
#for label 3: 1798 - 286 = 1512, floor(1512/286) = 5, 1512 - 5*286 = 82
#for label 4: 1798 - 236 = 1562, floor(1562/236) = 6, 1562 - 6*236 = 146
#for label 5: 1798 - 225 = 1573, floor(1573/225) = 6, 1573 - 6*225 = 223
#for label 6:  1798 - 294 = 1504, floor(1504/294) = 5, 1504 - 5*294 = 34
#for label 7: 1798 - 258 = 1540, floor(1540/258) = 5, 1540 - 5*258 = 250
#for label 8: 1798 - 253 = 1545, floor(1545/253) = 6, 1545 - 6*253 = 27
#for label 9: 1798 - 94 = 1704, floor(1704/94) = 18, 1704 - 18*94 = 12
#for label 10: 1798 - 89 = 1709, floor(1709/89) = 19, 1709 - 19*89 = 18
#for label 11: 1798 - 20 = 1778, floor(1778/20) = 88, 1778 - 88*20 = 18
#in total 17523


#####################################################################################################################################################
def augment_data(df, function):
    train_data = np.stack(df.sensor_data.to_numpy())
    df_np = df.to_numpy()

    augmentation = np.zeros(shape=(17523,128,6))
    whole_arr = np.zeros(shape=(17523,6), dtype=object)
    extra_counter_1 = 0
    extra_counter_2 = 0
    extra_counter_3 = 0
    extra_counter_4 = 0
    extra_counter_5 = 0
    extra_counter_6 = 0
    extra_counter_7 = 0
    extra_counter_8 = 0
    extra_counter_9 = 0
    extra_counter_10 = 0
    extra_counter_11 = 0
    k = 0
    for i in range (4053):
        #since 0 is the largest group (with 1798 samples), we do not want to augment the major group, just the data of the minor labels
        if df_np[i,1] == 1: 
            if extra_counter_1 < 160:
                for l in range (5+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_1 += 1
            else:
                for l in range (5): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 2: 
            if extra_counter_2 < 209:
                for l in range (6+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_2 += 1
            else:
                for l in range (6): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 3: 
            if extra_counter_3 < 82:
                for l in range (5+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_3 += 1
            else:
                for l in range (5): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 4: 
            if extra_counter_4 < 146:
                for l in range (6+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_4 += 1
            else:
                for l in range (6): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 5: 
            if extra_counter_5 < 223:
                for l in range (6+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_5 += 1
            else:
                for l in range (6): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 6: 
            if extra_counter_6 < 34:
                for l in range (5+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_6 += 1
            else:
                for l in range (5): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmantation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 7: 
            if extra_counter_7 < 250:
                for l in range (5+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_7 += 1
            else:
                for l in range (5): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 8: 
            if extra_counter_8 < 27:
                for l in range (6+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_8 += 1
            else:
                for l in range (6): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 9: 
            if extra_counter_9 < 12:
                for l in range (18+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_9 += 1
            else:
                for l in range (18): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 10: 
            if extra_counter_10 < 18:
                for l in range (19+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_10 += 1
            else:
                for l in range (19): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
        elif df_np[i,1] == 11: 
            if extra_counter_11 < 18:
                for l in range (88+1): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1
                extra_counter_11 += 1
            else:
                for l in range (88): 
                    augmentation[k,:,:] = function(train_data[i,:,:])
                    #copying the other data (label, model, velocity, mass, dec_average) without manipulating it:
                    whole_arr[k,1:5] = df_np[i,1:5]
                    #and the augmentation
                    whole_arr[k,0] = augmentation[k,:,:]
                    k += 1


    merged = np.append(df_np,whole_arr,axis = 0)

    columns = ['sensor_data', 'label', 'model', 'velocity', 'mass', 'deceleration_average']
    augmented_df = pd.DataFrame(merged, columns=columns)
    #convert it back into the DataFrame-framework
    return augmented_df

def NoiseAdding_all(df):
    return augment_data(df, NoiseAdding)

def AmplitudeScaling_all(df):
    return augment_data(df, AmplitudeScaling)

def ZoomIn_all(df):
    return augment_data(df, ZoomIn)

def Flip_all(df):
    return augment_data(df, Flip)

AUGMENTATION_METHODS = [lambda x:x, NoiseAdding_all, AmplitudeScaling_all, ZoomIn_all, Flip_all]
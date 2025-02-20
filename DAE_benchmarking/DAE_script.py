import pandas as p
import math
import numpy as np
import itertools as it
from haversine import *
from haversine_script import *
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,BatchNormalization
from keras.optimizers import Adam,Adamax
from scikeras.wrappers import KerasRegressor
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from scipy.stats import spearmanr
from collections import OrderedDict 
import time
import matplotlib.pyplot as plt
import json
from math import pi
import random



def calculate_DAE_Lemelson(X_train,Y_train,X_val,k,units = 'lat_lon'):
    """
    This function implements the method of Lemelson et al, 2009

    Parameters
    ----------
    X_train : 'numpy.ndarray'
        The training set's features (RSSI values)
    Y_train : 'numpy.ndarray'
        The training set's targets (locations)
    X_val : 'numpy.ndarray'
        The validation set's features (RSSI values)
    k : 'int'
        The number of the k nearest neighbors used in Lemelson's method
    units : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    DAE_val_Lemelson : 'list'
        The DAE estimates for the validation set, as estimated by the method of Lemelson et al, 2009
    """  
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
    distances, indices = nbrs.kneighbors(X_val)
    DAE_val_Lemelson = []
    val_length = X_val.shape[0]
    # print(val_length)
    for i in range(val_length):
        nearest = Y_train[indices[i,0],:]
        summing = 0
        for j in range(1,k): #should it be k+1?
            if units == 'lat_lon':
                summing = summing + haversine(tuple(nearest),tuple(Y_train[indices[i,j],:]))*1000
            else:
                summing = summing + np.linalg.norm(nearest-Y_train[indices[i,j],:])
        DAE_Lemelson = summing / (k-1) #should it be k?
        DAE_val_Lemelson.append(DAE_Lemelson)
    return DAE_val_Lemelson

    
def calculate_DAE_Marcus(X_train,Y_train,X_val,Y_M1_predict_in_val,k, units = 'lat_lon'):
    """
    This function implements the method of Marcus et al, 2013

    Parameters
    ----------
    X_train : 'numpy.ndarray'
        The training set's features (RSSI values)
    Y_train : 'numpy.ndarray'
        The training set's targets (locations)
    X_val : 'numpy.ndarray'
        The validation set's features (RSSI values)
    Y_M1_predict_in_val : 'numpy.ndarray'
        The predicted locations of the validation set's features, as provided by the positioning model M1
    k : 'int'
        The number of the k nearest neighbors used in Marcus' method
    units : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    DAE_val_Marcus : 'list'
        The DAE estimates for the validation set, as estimated by the method of Marcus et al, 2013
    """  
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
    distances, indices = nbrs.kneighbors(X_val)
    DAE_val_Marcus = []
    val_length = X_val.shape[0]
    # print(val_length)
    for i in range(val_length):
        location_estimate = Y_M1_predict_in_val[i,:]
        neighbors = []    
        dist = []  
        weights = []
        for j in range(k):
            neighbor = Y_train[indices[i,j],:] 
            neighbors.append(neighbor)            
            if units == 'lat_lon':
                dist.append(haversine(tuple(location_estimate),tuple(neighbor))*1000)  
            else:
                dist.append(np.linalg.norm(location_estimate-neighbor)) 
            weights.append(1/(1+distances[i,j]))  
        summing = 0 
        for j in range(k):
            summing = summing + dist[j] * weights[j]
        DAE_Marcus = summing / np.sum(weights)
        DAE_val_Marcus.append(DAE_Marcus)
    return DAE_val_Marcus



def calculate_DAE_Zou(X_train,Y_train,X_val,Y_M1_predict_in_val,k, method='mean',units = 'lat_lon'):
    """
    This function implements the method of Zou and Meng, 2014

    Parameters
    ----------
    X_train : 'numpy.ndarray'
        The training set's features (RSSI values)
    Y_train : 'numpy.ndarray'
        The training set's targets (locations)
    X_val : 'numpy.ndarray'
        The validation set's features (RSSI values)
    Y_M1_predict_in_val : 'numpy.ndarray'
        The predicted locations of the validation set's features, as provided by the positioning model M1
    k : 'int'
        The number of the k nearest neighbors used in Zou's method
    method : 'str'
        The selected variaty of the method of Zou and Meng, 2014, which can be "k", "mean", or "std"
    units : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    DAE_val_Zou : 'list'
        The DAE estimates for the validation set, as estimated by the method of Zou and Meng, 2014
    """ 
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
    distances, indices = nbrs.kneighbors(X_val)
    DAE_val_Zou = []
    val_length = X_val.shape[0]
    # print(val_length)
    for i in range(val_length):
        location_estimate = Y_M1_predict_in_val[i,:]
        neighbors = []    
        dist = []
        for j in range(1,k+1):
            neighbor = Y_train[indices[i,j-1],:] 
            neighbors.append(neighbor)     
            if units == 'lat_lon':
                dist.append(haversine(tuple(location_estimate),tuple(neighbor))*1000)   
            else:
                dist.append(np.linalg.norm(location_estimate-neighbor))   
        if method=='mean':
            DAE_Zou = np.mean(dist)
        elif method=='std':
            DAE_Zou = np.std(dist)            
        else:
            DAE_Zou = dist[k-1]  
        DAE_val_Zou.append(DAE_Zou)
    return DAE_val_Zou



def find_best_k_configuration_for_DAE(DAE_function,DAE_method_name,X_train,Y_train,X_val,Y_M1_predict_in_val, Y_M1_error_val,k_min,k_max,units = 'lat_lon'):
    """
    This function is finding best tuning for each rule-based DAE method

    Parameters
    ----------
    DAE_function : 'function'
        The function of the DAE method that is selected 
    DAE_method_name : 'str'
        The name of DAE method that is selected ('Lemelson','Marcus', or 'Zou')
    X_train : 'numpy.ndarray'
        The training set's features (RSSI values)
    Y_train : 'numpy.ndarray'
        The training set's targets (locations)
    X_val : 'numpy.ndarray'
        The validation set's features (RSSI values)
    Y_M1_predict_in_val : 'numpy.ndarray'
        The predicted locations of the validation set's features, as provided by the positioning model M1
    Y_M1_error_val : 'list'
        The errors of the predicted locations by the positioning model M1 on the validation set
    k_min : 'int'
        The minimum value of the range of k values to be explored
    k_max : 'int'
        The maximum value of the range of k values to be explored
    units : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    best_k : 'int'
        The value of k within the range (k_min,k_max) with the lowest DAE error
    error_list : 'list'
        The mean error of the studied method, for the various values of k in the range defined by (k_min,k_max)
    """     
    error_list = []
    best_k = k_min
    best_mean_error = 100000
    for k in range(k_min,k_max):
        print()
        print(k)
        if DAE_method_name == 'Lemelson':
            DAE_val = DAE_function(X_train,Y_train,X_val,k,units)
        elif DAE_method_name =='Marcus':
            DAE_val = DAE_function(X_train,Y_train,X_val,Y_M1_predict_in_val,k,units)
        elif DAE_method_name =='Zou':
            DAE_val = DAE_function(X_train,Y_train,X_val,Y_M1_predict_in_val,k,'mean',units)
        else:
            return None            
        DAE_miss_val = abs(np.asarray(Y_M1_error_val) - np.asarray(DAE_val))
        mean_error =  np.mean(DAE_miss_val)
        print(mean_error)
        error_list.append(np.mean(mean_error))
        if mean_error < best_mean_error:
            best_mean_error = mean_error
            best_k = k
    return best_k, error_list


def calculate_DAE_evaluation_metrics(M1_errors, DAE_estimates,DAE_method_name): 
    """
    This function calculates all relevant DAE evaluation metrics

    Parameters
    ----------
    M1_errors : 'list'
        The list with the errors of all estimates of the positioning model M1
    DAE_estimates : 'list'
        The list with all DAE estimates to be evaluated
    DAE_method_name : 'str'
        The name of DAE method that is evaluated

    Returns
    -------
    metrics : 'collections.OrderedDict'
        The ordered dictionary with all calculated evaluation metrics
    """  
    metrics = OrderedDict() 
    length = len(M1_errors)    
    DAE_miss_singed = (np.asarray(M1_errors).reshape((-1,)) - np.asarray(DAE_estimates))
    print("DAE_miss_singed", DAE_miss_singed.shape)
    DAE_miss = abs(np.asarray(M1_errors).reshape((-1,)) - np.asarray(DAE_estimates))   
    print("DAE_miss", DAE_miss.shape)
    in_circle_errors = [i for i in DAE_miss_singed if i < 0.0]
    out_circle_errors = [i for i in DAE_miss_singed if i > 0.0]
    metrics['method_name'] = DAE_method_name 
    metrics['mean'] = np.around(np.mean(DAE_miss),decimals=2)
    metrics['median'] = np.around(np.median(DAE_miss),decimals=2)
    metrics['75th'] = np.around(np.percentile(DAE_miss,75),decimals=2)
    metrics['90th'] = np.around(np.percentile(DAE_miss,90),decimals=2)
    metrics['std'] = np.around(np.std(DAE_miss),decimals=2)
    metrics['ov%'] =  np.around(100*len(in_circle_errors)/length,decimals=2)
    metrics['ov_md'] =  np.abs(np.around(np.median(in_circle_errors),decimals=2))
    metrics['ov_mn'] =  np.abs(np.around(np.mean(in_circle_errors),decimals=2))
    metrics['un_md'] = np.around(np.median(out_circle_errors),decimals=2)
    metrics['un_mn'] = np.around(np.mean(out_circle_errors),decimals=2)
    metrics['Pearson'] =  np.around(np.corrcoef(M1_errors, DAE_estimates)[0,1],decimals=4)
    Spearman, pi = spearmanr(M1_errors, DAE_estimates)
    metrics['Spearman'] = np.around(Spearman,decimals=4)
#     metrics['p-value'] = pi
    metrics
    return metrics




def calculate_DAE_baselines(y_M1_error_train_2, x_val, random_state=42): 
    """
    This function calculates several naive DAE baselines

    Parameters
    ----------
    y_M1_error_train_2 : 'list'
        The errors of the predicted locations by the positioning model M1 on the train_2 set
    x_val : 'numpy.ndarray'
        The validation set's features (RSSI values)
    random_state : 'int'
        The random seed, initializing the random number generation

    Returns
    -------
    DAE_constant_mean : 'list'
        The constant prediction of the DAE_constant_mean baseline method, taking as value the mean positioning error of the train_2 set
    DAE_constant_median : 'list'
        The constant prediction of the DAE_constant_median baseline method, taking as value the median positioning error of the train_2 set
    DAE_uniform_random : 'list'
        The uniformly random predictions of the DAE_uniform_random baseline method
    DAE_normal_random : 'numpy.ndarray'
        The normally distributed random predictions  of the DAE_normal_random baseline method
    """  
    val_length = x_val.shape[0]
    print(val_length)

    # Constant value, equal to the mean positioning error
    y_M1_error_train_2_mean = np.mean(y_M1_error_train_2) 
    DAE_constant_mean = [y_M1_error_train_2_mean for i in range(val_length)]


    # Constant value, equal to the median positioning error
    y_M1_error_train_2_median = np.median(y_M1_error_train_2) 
    DAE_constant_median = [y_M1_error_train_2_median for i in range(val_length)]


    # Radnom value, uniformly distributed in a range
    random.seed(random_state)
    y_M1_error_train_2_std = np.std(y_M1_error_train_2) 
    DAE_uniform_random = [random.randint(0, int(y_M1_error_train_2_mean)) for i in range(val_length)]

    # Radnom value, normaly distributed with mean and str, taken from the M1 performance
    np.random.seed(random_state)
    DAE_normal_random = abs(np.random.normal(np.mean(y_M1_error_train_2) , np.std(y_M1_error_train_2)/4 , val_length))

    return DAE_constant_mean, DAE_constant_median, DAE_uniform_random, DAE_normal_random

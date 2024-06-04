import pandas as pd
import numpy as np
import scipy as sp
import IPython
import sklearn
from haversine import haversine
from sklearn.model_selection import train_test_split
import statistics


def calculate_pairwise_error_list(ground_truth, predictions, reference="lat_lon"):
    """
    This function calculates the list of pairwise errors in meters between pairs of ground truth locations and their respective predictions.

    Parameters
    ----------
    ground_truth : 'numpy.ndarray'
        The ground truth locations
    predictions : 'numpy.ndarray'
        The predicted locations
    reference : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    distances : 'list'
        The list of pairwise error values
    """  
    distances = list()
    for i,_ in enumerate(predictions):
        if reference== "lat_lon":
            ground_truth_list = ground_truth[i].tolist()
            predict_list = predictions[i].tolist()
            h= haversine(tuple(ground_truth_list),tuple(predict_list))*1000  # multiplying by 1000 to transform from Km to m
            distances.append(h)
        else:
            distances.append(np.linalg.norm(ground_truth[i,:]-predictions[i,:]))
    return distances

def my_custom_mean_error(ground_truth, predictions, reference="lat_lon"):
    """
    This function calculates the mean localization error.

    Parameters
    ----------
    ground_truth : 'numpy.ndarray'
        The ground truth locations
    predictions : 'numpy.ndarray'
        The predicted locations
    reference : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    mean_error : 'list'
        The mean positioning error
    """  
    distances = calculate_pairwise_error_list(ground_truth, predictions, reference)
    return statistics.mean(distances)


def my_custom_error_stats(ground_truth, predictions,statistical_metric='mean',percentile=50, reference = "lat_lon"):
    """
    This function calculates the selected statistics of localization error.

    Parameters
    ----------
    ground_truth : 'numpy.ndarray'
        The ground truth locations
    predictions : 'numpy.ndarray'
        The predicted locations
    statistical_metric : 'str'
        Can be either 'mean', 'median' or 'percentile', depending on the desired statistical metric
    percentile : 'int'
        The percentile of error to be returned, if statistical_metric=='percentile'
    reference : 'str'
        Can be either 'lat_lon' or 'meters', depending on whether the coordinates are in latitude/longitude or in a relative reference system in meters. 

    Returns
    -------
    returns : 'int'
        The desired statistical metric
    """  
    distances = calculate_pairwise_error_list(ground_truth, predictions, reference)
    if statistical_metric=="mean":
        return statistics.mean(distances)
    elif statistical_metric=="median":
        return statistics.median(distances)	
    elif statistical_metric=="percentile" and (percentile>=0 or percentile<=100):
        return np.percentile(distances,percentile)	
    else:
        return statistics.mean(distances)

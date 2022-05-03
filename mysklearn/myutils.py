"""PA7
Name: Luke Martin
Clss: CPSC 322 - 02
Sems: Spring 2022
Date: April 12, 2022

mysklearn.myutils.py
Description:
    This python file contains the generic functions used throughout the
    mysklearn module and for PA7.
"""

from copy import deepcopy
from math import sqrt

def get_frequencies(data):
    """Returns a dictionary where the keys are the unique values from data
    and the values are the frequencies that the values are seen.

    Args:
        data(list): list to count the frequency of values in

    Returns:
        dict of ints: the number of times each unique value for col_name is seen in the table.
    """
    frequencies = {}
    for val in data:
        if val in frequencies.keys():
            frequencies[val] += 1
        else:
            frequencies[val] = 1
    return frequencies


def dist(x_1, x_2):
    """Calculates the distance between points/instances x1 and x2.
    For categorical values, we just take the distance between them to be 1
    if not the same, and 0 if they are.

    Args:
        x1(list): list of numerics/labels
        x2(list): list of numerics/labels

    Returns:
        float: the euclidean distance between x1 and x2 in n-space
    """
    radicand = 0
    for i, _ in enumerate(x_1):
        if isinstance(x_1[i], str) or isinstance(x_2[i], str):
            if x_1[i] != x_2[i]:
                radicand += 1
        else:
            radicand += (x_1[i] - x_2[i]) ** 2

    return sqrt(radicand)


def normalize_data(x):
    """Normalizes the 1d data stored in x

    Args:
        x(list): list of numerics

    Returns:
        list: the same data but normalized on a 0-1 scale
    """
    normal_data = [(x[i] - min(x))/(max(x) - min(x)) for i in range(len(x))]
    return normal_data


def get_indicies_per_class(y, index_list):
    """Creates a dictionary of labels and their index locations.
    Keys are unique labels in y, values are the list of indexes where
    they are in the list.

    Args:
        y(list of labels): a 1D labels list
        index_list(list of ints): a list of the index values, potentially shuffled

    Returns:
        label_indicies(dictionary of lists): the index locations of each label
    """
    label_indicies = {}
    for index in index_list:
        label = y[index]
        if label in label_indicies.keys():
            label_indicies[label].append(index)
        else:
            label_indicies[label] = [index]
    return label_indicies


def groupby(data, col_index):
    """Return a dictionary of 2D lists where the keys are the unique values
    from col_index and the values are the instances
    containing those values in col_index position.

    Args:
        data(list of lists): 2D list representing a table of data
        col_index(int): index of column to groupby.

    Returns:
        dict of list of lists: the instances that contain each unique value for col_index.
    """
    groupby_data = {}

    for row in data:
        if row[col_index] in groupby_data.keys():
            groupby_data[row[col_index]].append(deepcopy(row))
        else:
            groupby_data[row[col_index]] = [deepcopy(row)]

    return groupby_data

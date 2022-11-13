import time
import os
from collections import OrderedDict

import numpy as np
import pandas as pd


def path_decorator(path):
    if not path[-1] == "/" and not path[-1] == "\\":
        new_path = path + "/"
    else: 
        new_path = path
    return new_path


def datetime_dir(
    save_dir="./",
    dir_suffix=None,
):
    save_dir = path_decorator(save_dir)

    current_time = time.localtime()
    current_month_dir = save_dir + time.strftime("%h/", current_time)
    current_date_dir = current_month_dir + \
        time.strftime("%d_%H-%M", current_time)

    if dir_suffix != "" and dir_suffix is not None:
        current_date_dir = current_date_dir + "_" + dir_suffix + "/"
    else:
        current_date_dir = current_date_dir + "/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(current_month_dir):
        os.mkdir(current_month_dir)
    if not os.path.exists(current_date_dir):
        os.mkdir(current_date_dir)

    print(f"Current save directory: {current_date_dir}")
    return current_date_dir


def save_variable_dict(file_name, variable_dict):
    new_dict = dict([(key, [val]) for key, val in variable_dict.items()])
    pd.DataFrame.from_dict(
        new_dict,
        orient="columns",
    ).to_csv(file_name)

def load_variable_dict(file_name):
    list_dict = pd.read_csv(
        file_name, 
        index_col=0,
        header=0
    ).to_dict(orient='list')
    new_dict = dict([(key, val[0]) for key, val in list_dict.items()])
    return new_dict


def save_variable_list_dict(file_name, variable_list_dict, orient='columns'):
    """
    orient = 'index' is always used when variable list are not equal in length
    """
    pd.DataFrame.from_dict(
        variable_list_dict,
        orient=orient,
    ).to_csv(file_name)


def load_variable_list_dict(file_name, throw_nan=True, orient='columns'):
    """
    orient = 'index' should be used when variable list are not equal in length
    """
    if orient == 'index':
        variable_list_dict = pd.read_csv(
            file_name, index_col=0, header=0).transpose().to_dict(orient='list')
    elif orient == 'columns':
        variable_list_dict = pd.read_csv(
            file_name, index_col=0, header=0).to_dict(orient='list')
    else:
        raise ValueError("only recognize 'index' or 'columns' for orient")

    if not throw_nan:
        return OrderedDict([(key, np.array(val)) for key, val in variable_list_dict.items()])

    for key, val in variable_list_dict.items():
        new_val = np.array(val)
        new_val = new_val[~np.isnan(val)]
        variable_list_dict[key] = new_val
    return OrderedDict(variable_list_dict)

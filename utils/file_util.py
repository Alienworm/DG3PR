import os
import dill
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def save_to_dill(data: object, file_path: str) -> None:
    """
    save data to dill file
    :param data: dataframe
    :param file_path: save path
    :return: None
    """

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, 'wb') as f:
        dill.dump(data, f)


def load_from_dill(file_path: str) -> object:
    """
    load data from dill file
    :param file_path: load path
    :return: data, data type
    """

    with open(file_path, 'rb') as f:
        data = dill.load(f)
    return data


def load_csv_file(file_path: str) -> pd.DataFrame:
    """
    load csv file
    :param file_path: file path
    :return: dataframe
    """

    if file_path.endswith('.dill'):
        data = load_from_dill(file_path)
    else:
        data = pd.read_csv(file_path)

    return data
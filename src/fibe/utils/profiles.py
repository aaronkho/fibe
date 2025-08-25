import copy
import logging
from pathlib import Path
from typing import Any, Final, Self
from collections.abc import MutableMapping, Mapping, MutableSequence, Sequence, Iterable
import numpy as np
import pandas as pd
import xarray as xr


logger = logging.getLogger('fibe')
logger.setLevel(logging.INFO)

array_types = (list, tuple, np.ndarray)


def read_profiles_file(fname, interface='xarray'):
    if interface == 'pandas':
        return read_profiles_file_pandas(fname)
    elif interface == 'ascii':
        return read_profiles_file_ascii(fname)
    else:
        return read_profiles_file_xarray(fname)


def read_profiles_file_xarray(fname):
    ds = xr.open_dataset(fname)
    return _get_profiles_data_from_dataframe_for_dataset(ds)


def read_profiles_file_pandas(fname):
    df = pd.read_hdf(fname, key='/data')
    return _get_profiles_data_from_dataframe_for_dataset(df)


def read_profiles_file_ascii(fname):
    df = pd.read_csv(fname, delimiter=' ', header=0)
    return _get_profiles_data_from_dataframe_for_dataset(df)


def _get_profiles_data_from_dataframe_for_dataset(data):
    profiles = {}
    if isinstance(data, (pd.DataFrame, xr.Dataset, xr.DataTree)):
        if 'psin' in data:
            profiles['psinorm'] = data['psin'].to_numpy().flatten()
        elif 'psinorm' in data:
            profiles['psinorm'] = data['psinorm'].to_numpy().flatten()
        elif 'xpsi' in data:
            profiles['psinorm'] = data['xpsi'].to_numpy().flatten()
        if 'fpol' in data:
            profiles['fpol'] = data['fpol'].to_numpy().flatten()
        elif 'f' in data:
            profiles['fpol'] = data['f'].to_numpy().flatten()
        if 'pres' in data:
            profiles['pres'] = data['pres'].to_numpy().flatten()
        elif 'pressure' in data:
            profiles['pres'] = data['pressure'].to_numpy().flatten()
        elif 'p' in data:
            profiles['pres'] = data['p'].to_numpy().flatten()
        if 'qpsi' in data:
            profiles['qpsi'] = data['qpsi'].to_numpy().flatten()
        elif 'q' in data:
            profiles['qpsi'] = data['q'].to_numpy().flatten()
    if profiles and 'psinorm' not in profiles:
        profiles['psinorm'] = None
    return profiles


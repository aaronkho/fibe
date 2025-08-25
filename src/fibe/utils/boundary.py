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


def read_boundary_file(fname, interface='xarray'):
    if interface == 'eqdsk':
        return read_boundary_file_geqdsk_like(fname)
    elif interface == 'ascii':
        return read_boundary_file_ascii(fname)
    else:
        return read_boundary_file_xarray(fname)


def read_boundary_file_xarray(fname):
    ds = xr.open_dataset(fname)
    return _get_boundary_data_from_dataframe_or_dataset(ds)


def read_boundary_file_pandas(fname):
    df = pd.read_hdf(fname, key='/data')
    return _get_boundary_data_from_dataframe_or_dataset(df)


def read_boundary_file_ascii(fname):
    df = pd.read_csv(fname, delimiter=' ', header=0)
    return _get_boundary_data_from_dataframe_or_dataset(df)


def read_boundary_file_geqdsk_like(fname):

    def _sep_eq_line(line, float_width=16, floats_per_line=5, sep=' '):
        """ Split a eqdsk-style line and inserts seperator characters """
        splitted = [
            line[num*float_width:(num+1)*float_width] for num in range(floats_per_line)
        ]
        separate = sep.join(splitted)
        return separate

    def _read_chunk(lines, length, floats_per_line=5):
        """ Read a single chunk (array/vector)

        Reads and pops for `lines` the amount of lines
        containing the to be read vector.

        Args:
            lines:  List of lines to be read. Destructive!
            length: Length of to be read vector

        Kwargs:
            floats_per_line: Amount of floats on a line [Default: 5]
        """
        num_lines = int(np.ceil(length / floats_per_line))
        vals = []
        for line in lines[:num_lines]:
            sep = _sep_eq_line(line)
            vals.append(np.fromstring(sep, sep=' '))
        del lines[:num_lines]
        return vals

    with open(fname, 'r') as ff:
        lines = ff.readlines()

    # Read boundary vector, GEQDSK-style
    boundary = {}
    header = lines.pop(0)
    boundary['nbdry'] = int(header[:5])
    if boundary['nbdry'] > 0:
        bdry = _read_chunk(lines, eq['nbdry'] * 2)
        bdry = np.hstack(bdry).reshape((eq['nbdry'], 2))
        boundary['rbdry'] = bdry[:, 0]
        boundary['zbdry'] = bdry[:, 1]
    else:
        boundary['rbdry'] = np.array([])
        boundary['zbdry'] = np.array([])

    return boundary


def _get_boundary_data_from_dataframe_or_dataset(data):
    boundary = {}
    if isinstance(data, (pd.DataFrame, xr.Dataset, xr.DataTree)):
        if 'rbdry' in data and 'zbdry' in data:
            boundary['rbdry'] = data['rbdry'].to_numpy().flatten()
            boundary['zbdry'] = data['zbdry'].to_numpy().flatten()
        elif 'xbdry' in data and 'zbdry' in data:
            boundary['rbdry'] = data['xbdry'].to_numpy().flatten()
            boundary['zbdry'] = data['zbdry'].to_numpy().flatten()
        elif 'rbbbs' in data and 'zbbbs' in data:
            boundary['rbdry'] = data['rbbbs'].to_numpy().flatten()
            boundary['zbdry'] = data['zbbbs'].to_numpy().flatten()
        elif 'r' in data and 'z' in data:
            boundary['rbdry'] = data['r'].to_numpy().flatten()
            boundary['zbdry'] = data['z'].to_numpy().flatten()
        elif 'x' in data and 'z' in data:
            boundary['rbdry'] = data['x'].to_numpy().flatten()
            boundary['zbdry'] = data['z'].to_numpy().flatten()
        elif 'x' in data and 'y' in data:
            boundary['rbdry'] = data['x'].to_numpy().flatten()
            boundary['zbdry'] = data['y'].to_numpy().flatten()
        if 'rbdry' in boundary:
            boundary['nbdry'] = len(boundary['rbdry'])
    return boundary


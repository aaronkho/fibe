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
        if 'jstar' in data:
            profiles['jstar'] = data['jstar'].to_numpy().flatten()
        # Kinetic profiles (electron density [m^-3], electron temperature
        # [eV]) -- an alternative to a directly-supplied 'pres' column, for
        # callers that only have diagnostic ne/Te measurements rather than an
        # already-integrated pressure. Use compute_pressure_from_kinetic_
        # profiles to turn these into 'pres' before handing them to
        # FixedBoundaryEquilibrium.define_pressure_profile.
        if 'ne' in data:
            profiles['ne'] = data['ne'].to_numpy().flatten()
        elif 'n_e' in data:
            profiles['ne'] = data['n_e'].to_numpy().flatten()
        if 'te' in data:
            profiles['te'] = data['te'].to_numpy().flatten()
        elif 't_e' in data:
            profiles['te'] = data['t_e'].to_numpy().flatten()
    if profiles and 'psinorm' not in profiles:
        profiles['psinorm'] = None
    return profiles


ELEMENTARY_CHARGE = 1.602176634e-19  # C (== J/eV)


def compute_pressure_from_kinetic_profiles(ne, te, ni_ratio=1.0, ti_ratio=1.0):
    '''Returns the total (electron + ion) kinetic pressure [Pa] implied by
    an electron density `ne` [m^-3] and temperature `te` [eV] profile, under
    the assumption n_i = ni_ratio * n_e and T_i = ti_ratio * T_e:

        p = e * (n_e * T_e + n_i * T_i) = e * n_e * T_e * (1 + ni_ratio * ti_ratio)

    The defaults (ni_ratio=ti_ratio=1.0, i.e. n_i=n_e, T_i=T_e) are a
    generic, device-agnostic assumption -- override with real ion
    density/temperature ratios (e.g. accounting for main-ion dilution by
    impurities, or a measured Ti/Te ratio) where available.
    '''
    ne = np.asarray(ne, dtype=float)
    te = np.asarray(te, dtype=float)
    ni = ni_ratio * ne
    ti = ti_ratio * te
    return ELEMENTARY_CHARGE * (ne * te + ni * ti)


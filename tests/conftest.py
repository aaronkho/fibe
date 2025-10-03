import pytest
from pathlib import Path
from fibe import FixedBoundaryEquilibrium


@pytest.fixture(scope='session')
def geqdsk_file_path():
    yield Path(__file__).parent / 'data' / 'test_input.geqdsk'


@pytest.fixture(scope='session')
def scratch_grid():
    yield {
        'nr': 129,
        'nz': 129,
        'rmin': 1.5,
        'rmax': 4.5,
        'zmin': -2.0,
        'zmax': 2.0,
    }


@pytest.fixture(scope='session')
def scratch_mxh_boundary():
    yield {
        'rgeo': 3.0,
        'zgeo': 0.0,
        'rminor': 1.0,
        'kappa': 1.5,
        'cos_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'sin_coeffs': [0.0, 0.5, -0.1, 0.0, 0.0, 0.0, 0.0],
    }


@pytest.fixture(scope='session')
def scratch_fp_profiles():
    yield {
        'psinorm': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'f': [10.0, 9.99, 9.98, 9.97, 9.96, 9.95],
        'pressure': [1.0e6, 9.0e5, 7.0e5, 4.0e5, 2.0e5, 1.0e5],
    }


@pytest.fixture(scope='session')
def regrid_specs():
    yield {
        'nr': 129,
        'nz': 129,
        'optimal': True,
    }


@pytest.fixture(scope='class')
def empty_class():
    yield FixedBoundaryEquilibrium()

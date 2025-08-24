import pytest
import numpy as np
from fibe import FixedBoundaryEquilibrium


@pytest.mark.usefixtures('geqdsk_file_path')
class TestCreation():

    def test_creation_empty(self):
        eq = FixedBoundaryEquilibrium()
        assert isinstance(eq, FixedBoundaryEquilibrium)
        assert (not eq._data)

    def test_creation_with_geqdsk(self, geqdsk_file_path):
        eq = FixedBoundaryEquilibrium.from_geqdsk(geqdsk_file_path)
        assert isinstance(eq, FixedBoundaryEquilibrium)
        assert eq._data.get('nr', None) == 61
        assert eq._data.get('nz', None) == 129
        assert eq._data.get('nbdry', None) == 32 + 1  # Add 1 due to enforcing closed boundary vector
        assert eq._data.get('nlim', None) == 0


@pytest.mark.usefixtures('empty_class', 'scratch_grid', 'scratch_mxh_boundary', 'scratch_fp_profiles')
class TestInitializationWithFP():

    def test_grid_initialization(self, empty_class, scratch_grid):
        empty_class.define_grid(**scratch_grid)
        assert 'nr' in empty_class._data
        assert 'nz' in empty_class._data
        assert 'rleft' in empty_class._data
        assert 'rdim' in empty_class._data
        assert 'zmid' in empty_class._data
        assert 'zdim' in empty_class._data

    def test_mxh_boundary_initialization(self, empty_class, scratch_mxh_boundary):
        empty_class.define_boundary_with_mxh(**scratch_mxh_boundary)
        assert 'nbdry' in empty_class._data
        assert 'rbdry' in empty_class._data
        assert 'zbdry' in empty_class._data

    def test_fp_profiles_initialization(self, empty_class, scratch_fp_profiles):
        empty_class.define_f_and_pressure_profiles(**scratch_fp_profiles)
        assert 'fpol' in empty_class._data
        assert 'ffprime' in empty_class._data
        assert 'pres' in empty_class._data
        assert 'pprime' in empty_class._data
        assert 'fpol_fs' in empty_class._fit
        assert 'pres_fs' in empty_class._fit

    def test_psi_initialization(self, empty_class):
        empty_class.initialize_psi()
        assert 'rvec' in empty_class._data
        assert 'zvec' in empty_class._data
        assert 'psi' in empty_class._data
        assert 'simagx' in empty_class._data
        assert 'sibdry' in empty_class._data
        assert 'cur' in empty_class._data
        assert 'curscale' in empty_class._data
        assert 'cpasma' in empty_class._data

    def test_psi_solver(self, empty_class):
        empty_class.solve_psi()
        assert 'qpsi' in empty_class._data
        assert 'gcase' in empty_class._data
        assert 'gid' in empty_class._data


@pytest.mark.usefixtures('empty_class', 'geqdsk_file_path')
class TestInitializationWithGEQDSK():

    def test_geqdsk_load(self, empty_class, geqdsk_file_path):
        empty_class.load_geqdsk(geqdsk_file_path)
        assert empty_class._data.get('nr', None) == 61
        assert empty_class._data.get('nz', None) == 129
        assert empty_class._data.get('nbdry', None) == 32 + 1  # Add 1 due to enforcing closed boundary vector
        assert empty_class._data.get('nlim', None) == 0

    def test_psi_regrid(self, empty_class, regrid_specs):
        empty_class.regrid(**regrid_specs)
        assert empty_class._data.get('nr', None) == 129
        assert empty_class._data.get('nz', None) == 129


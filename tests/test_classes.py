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


@pytest.mark.usefixtures('scratch_grid', 'scratch_mxh_boundary', 'scratch_fp_profiles')
class TestInitializationWithFP():

    eq = FixedBoundaryEquilibrium()

    def test_grid_initialization(self, scratch_grid):
        self.eq.define_grid(**scratch_grid)
        assert 'nr' in self.eq._data
        assert 'nz' in self.eq._data
        assert 'rleft' in self.eq._data
        assert 'rdim' in self.eq._data
        assert 'zmid' in self.eq._data
        assert 'zdim' in self.eq._data

    def test_mxh_boundary_initialization(self, scratch_mxh_boundary):
        self.eq.define_boundary_with_mxh(**scratch_mxh_boundary)
        assert 'nbdry' in self.eq._data
        assert 'rbdry' in self.eq._data
        assert 'zbdry' in self.eq._data

    def test_fp_profiles_initialization(self, scratch_fp_profiles):
        self.eq.define_f_and_pressure_profiles(**scratch_fp_profiles)
        assert 'fpol' in self.eq._data
        assert 'ffprime' in self.eq._data
        assert 'pres' in self.eq._data
        assert 'pprime' in self.eq._data
        assert 'fpol_fs' in self.eq._fit
        assert 'pres_fs' in self.eq._fit

    def test_psi_initialization(self):
        self.eq.initialize_psi()
        assert 'rvec' in self.eq._data
        assert 'zvec' in self.eq._data
        assert 'psi' in self.eq._data
        assert 'simagx' in self.eq._data
        assert 'sibdry' in self.eq._data
        assert 'cur' in self.eq._data
        assert 'curscale' in self.eq._data
        assert 'cpasma' in self.eq._data

    def test_psi_solver(self):
        self.eq.solve_psi()
        assert 'qpsi' in self.eq._data
        assert 'psi_error' in self.eq._data
        assert 'errsol' in self.eq._data


@pytest.mark.usefixtures('geqdsk_file_path')
class TestInitializationWithGEQDSK():

    eq = FixedBoundaryEquilibrium()

    def test_geqdsk_load(self, geqdsk_file_path):
        self.eq.load_geqdsk(geqdsk_file_path)
        assert self.eq._data.get('nr', None) == 61
        assert self.eq._data.get('nz', None) == 129
        assert self.eq._data.get('nbdry', None) == 32 + 1  # Add 1 due to enforcing closed boundary vector
        assert self.eq._data.get('nlim', None) == 0

    def test_psi_regrid(self, regrid_specs):
        self.eq.regrid(**regrid_specs)
        assert self.eq._data.get('nr', None) == 129
        assert self.eq._data.get('nz', None) == 129


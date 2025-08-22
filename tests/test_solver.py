import pytest
import numpy as np
from fibe.solver import FixedBoundaryEquilibrium



@pytest.mark.usefixtures('empty_class')
class TestCreation():


    def test_empty_class_creation(self, empty_class):
        assert isinstance(empty_class, FixedBoundaryEquilibrium)



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
        assert 'cur' in empty_class._data
        assert 'curscale' in empty_class._data
        assert 'cpasma' in empty_class._data

    def test_psi_initialization(self, empty_class):
        print(empty_class._data['xpsi'])
        empty_class.solve_psi()
        assert 'qpsi' in empty_class._data
        assert 'gcase' in empty_class._data
        assert 'gid' in empty_class._data


"""
Test suite for TDLM library
"""

import pytest
import numpy as np
import pandas as pd
from TDLM import tdlm, TDLMError


class TestTDLM:
    """Test class for TDLM functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.n = 3
        self.mi = np.array([100, 200, 150])
        self.mj = np.array([80, 180, 120])
        self.dij = np.array([[0, 10, 15], [10, 0, 8], [15, 8, 0]])
        self.sij = np.array([[0, 5, 7], [5, 0, 4], [7, 4, 0]])
        self.Oi = np.array([50, 80, 60])
        self.Dj = np.array([40, 90, 50])
        self.Tij = np.array([[0, 25, 25], [30, 0, 50], [35, 35, 0]])
    
    def test_run_law_model_basic(self):
        """Test basic functionality of run_law_model"""
        result = tdlm.run_law_model(
            law='NGravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            repli=1
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, self.n, self.n)
        assert np.all(result >= 0)
    
    def test_run_law_model_multiple_exponents(self):
        """Test run_law_model with multiple exponents"""
        exponents = [0.1, 0.5, 1.0]
        result = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=exponents,
            model='PCM',
            out_trips=self.Oi,
            repli=2
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(exponents)
        for exp in exponents:
            assert exp in result
            assert result[exp].shape == (2, self.n, self.n)
    
    def test_invalid_law(self):
        """Test error handling for invalid law"""
        with pytest.raises(TDLMError):
            tdlm.run_law_model(
                law='InvalidLaw',
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                exponent=0.5
            )
    
    def test_invalid_model(self):
        """Test error handling for invalid model"""  
        with pytest.raises(TDLMError):
            tdlm.run_law_model(
                law='GravExp',
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                exponent=0.5,
                model='InvalidModel'
            )
    
    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatch"""
        with pytest.raises(TDLMError):
            tdlm.run_law_model(
                law='GravExp',
                mass_origin=self.mi,
                mass_destination=np.array([80, 180]),  # Wrong size
                distance=self.dij,
                exponent=0.5
            )
    
    def test_opportunity_laws(self):
        """Test laws that require opportunity matrix"""
        for law in ['Rad', 'RadExt', 'Schneider']:
            result = tdlm.run_law_model(
                law=law,
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                opportunity=self.sij,
                exponent=0.5,
                model='UM',
                out_trips=self.Oi,
                repli=1
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, self.n, self.n)
    
    def test_constrained_models(self):
        """Test constrained models"""
        for model in ['PCM', 'ACM', 'DCM']:
            kwargs = {'out_trips': self.Oi}
            if model in ['ACM', 'DCM']:
                kwargs['in_trips'] = self.Dj
                
            result = tdlm.run_law_model(
                law='GravExp',
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                exponent=0.5,
                model=model,
                repli=1,
                **kwargs
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, self.n, self.n)
    
    def test_gof_single_simulation(self):
        """Test goodness-of-fit for single simulation"""
        sim = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            repli=3
        )
        
        gof_result = tdlm.gof(sim=sim, obs=self.Tij, distance=self.dij)
        
        assert isinstance(gof_result, pd.DataFrame)
        assert len(gof_result) == 3  # 3 replications
        assert 'CPC' in gof_result.columns
        assert 'RMSE' in gof_result.columns
    
    def test_gof_multiple_simulations(self):
        """Test goodness-of-fit for multiple simulations"""
        exponents = [0.1, 0.5]
        sim = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=exponents,
            model='UM',
            out_trips=self.Oi,
            repli=2
        )
        
        gof_result = tdlm.gof(sim=sim, obs=self.Tij, distance=self.dij)
        
        assert isinstance(gof_result, dict)
        assert len(gof_result) == len(exponents)
        for exp in exponents:
            assert exp in gof_result
            assert isinstance(gof_result[exp], pd.DataFrame)
    
    def test_gof_specific_measures(self):
        """Test goodness-of-fit with specific measures"""
        sim = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            repli=1
        )
        
        measures = ['CPC', 'RMSE']
        gof_result = tdlm.gof(sim=sim, obs=self.Tij, distance=self.dij, measures=measures)
        
        assert isinstance(gof_result, pd.DataFrame)
        assert all(col in gof_result.columns for col in measures)
        assert len([col for col in gof_result.columns if col not in measures + ['Replication']]) == 0
    
    def test_return_proba(self):
        """Test probability matrix output"""
        result = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            repli=1,
            return_proba=True
        )
        
        assert isinstance(result, dict)
        assert 'simulations' in result
        assert 'probabilities' in result
        assert result['simulations'].shape == (1, self.n, self.n)
        assert result['probabilities'].shape == (self.n, self.n)


if __name__ == '__main__':
    pytest.main([__file__])

"""
Test suite for TDLM library
"""

import pytest
import numpy as np
import pandas as pd
from TDLM import tdlm, _TDLMError


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
        with pytest.raises(_TDLMError):
            tdlm.run_law_model(
                law='InvalidLaw',
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                exponent=0.5
            )
    
    def test_invalid_model(self):
        """Test error handling for invalid model"""  
        with pytest.raises(_TDLMError):
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
        with pytest.raises(_TDLMError):
            tdlm.run_law_model(
                law='GravExp',
                mass_origin=self.mi,
                mass_destination=np.array([80, 180]),  # Wrong size
                distance=self.dij,
                exponent=0.5
            )
    
    def test_opportunities_laws(self):
        """Test laws that require opportunity matrix"""
        for law in ['Rad', 'RadExt', 'Schneider']:
            result = tdlm.run_law_model(
                law=law,
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                opportunities=self.sij,
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

    def test_run_law_model_average_mode(self):
        """Test run_law_model with average=True"""
        result = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            average=True
        )
        
        assert isinstance(result, np.ndarray)
        # When average=True, repli is forced to 1
        assert result.shape == (1, self.n, self.n)
        assert np.all(result >= 0)
    
    def test_run_law_model_average_mode_multiple_exponents(self):
        """Test run_law_model with average=True and multiple exponents"""
        exponents = [0.1, 0.5, 1.0]
        result = tdlm.run_law_model(
            law='NGravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=exponents,
            model='PCM',
            out_trips=self.Oi,
            average=True,
            repli=10  # This should be ignored when average=True
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(exponents)
        for exp in exponents:
            assert exp in result
            # Should be (1, n, n) even though repli=10 was specified
            assert result[exp].shape == (1, self.n, self.n)
    
    def test_run_law_model_average_vs_replications(self):
        """Test that average=True produces deterministic results unlike multiple replications"""
        # Run with average mode
        result_avg = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            average=True,
            random_seed=42
        )
        
        # Run again with same seed - should be identical
        result_avg2 = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            average=True,
            random_seed=42
        )
        
        np.testing.assert_array_equal(result_avg, result_avg2)
        
        # Run with multiple replications - individual replications will differ
        result_repli = tdlm.run_law_model(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            model='UM',
            out_trips=self.Oi,
            average=False,
            repli=3,
            random_seed=42
        )
        
        assert result_repli.shape == (3, self.n, self.n)
        # Check that replications differ from each other (probabilistic test)
        assert not np.array_equal(result_repli[0], result_repli[1])
    
    def test_run_law_model_average_all_models(self):
        """Test average mode works with all constrained models"""
        for model in ['UM', 'PCM', 'ACM', 'DCM']:
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
                average=True,
                **kwargs
            )
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, self.n, self.n)
            assert np.all(result >= 0)
            
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

    def test_run_law_basic(self):
        """Test basic functionality of run_law"""
        result = tdlm.run_law(
            law='NGravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.n, self.n)
        assert np.all(result >= 0)
        # Check diagonal is zero
        assert np.all(np.diag(result) == 0)

    def test_run_law_multiple_exponents(self):
        """Test run_law with multiple exponents"""
        exponents = [0.1, 0.5, 1.0]
        result = tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=exponents
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(exponents)
        for exp in exponents:
            assert exp in result
            assert result[exp].shape == (self.n, self.n)
            assert np.all(result[exp] >= 0)
    
    def test_run_law_all_laws(self):
        """Test run_law with all available laws"""
        laws_without_opportunities = ['GravExp', 'NGravExp', 'GravPow', 'NGravPow', 'Rand']
        
        for law in laws_without_opportunities:
            result = tdlm.run_law(
                law=law,
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                exponent=0.5
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (self.n, self.n)
    
    def test_run_law_with_opportunities(self):
        """Test run_law with laws requiring opportunities matrix"""
        laws_with_opportunities = ['Rad', 'RadExt', 'Schneider']
        
        for law in laws_with_opportunities:
            result = tdlm.run_law(
                law=law,
                mass_origin=self.mi,
                mass_destination=self.mj,
                distance=self.dij,
                opportunities=self.sij,
                exponent=0.5
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (self.n, self.n)
    
    def test_run_law_auto_compute_opportunities(self):
        """Test that run_law automatically computes opportunities when needed"""
        result = tdlm.run_law(
            law='Rad',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            processes=1  # Use single process for test
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.n, self.n)
    
    def test_run_model_basic(self):
        """Test basic functionality of run_model"""
        # First generate probabilities
        probabilities = tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5
        )
        
        # Wrap in dict as expected by run_model
        prob_dict = {0.5: probabilities}
        
        result = tdlm.run_model(
            probabilities=prob_dict,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='UM',
            out_trips=self.Oi,
            repli=1
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, self.n, self.n)
        assert np.all(result >= 0)
    
    def test_run_model_multiple_exponents(self):
        """Test run_model with multiple probability matrices"""
        exponents = [0.1, 0.5, 1.0]
        probabilities = tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=exponents
        )
        
        result = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='PCM',
            out_trips=self.Oi,
            repli=2
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(exponents)
        for exp in exponents:
            assert exp in result
            assert result[exp].shape == (2, self.n, self.n)
    
    def test_run_model_all_models(self):
        """Test run_model with all available models"""
        probabilities = {0.5: tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5
        )}
        
        # Test UM
        result = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='UM',
            out_trips=self.Oi,
            repli=1
        )
        assert isinstance(result, np.ndarray)
        
        # Test PCM
        result = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='PCM',
            out_trips=self.Oi,
            repli=1
        )
        assert isinstance(result, np.ndarray)
        
        # Test ACM
        result = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='ACM',
            out_trips=self.Oi,
            in_trips=self.Dj,
            repli=1
        )
        assert isinstance(result, np.ndarray)
        
        # Test DCM
        result = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='DCM',
            out_trips=self.Oi,
            in_trips=self.Dj,
            repli=1
        )
        assert isinstance(result, np.ndarray)
    
    def test_run_model_average_mode(self):
        """Test run_model with average=True"""
        probabilities = {0.5: tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5
        )}
        
        result = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='UM',
            out_trips=self.Oi,
            average=True
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, self.n, self.n)
    
    def test_run_law_and_run_model_pipeline(self):
        """Test the complete pipeline: run_law -> run_model -> gof"""
        # Step 1: Estimate probabilities
        exponents = [0.1, 0.5]
        probabilities = tdlm.run_law(
            law='NGravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=exponents
        )
        
        # Step 2: Run model
        simulations = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='PCM',
            out_trips=self.Oi,
            repli=2
        )
        
        # Step 3: Calculate GOF
        gof_result = tdlm.gof(sim=simulations, obs=self.Tij, distance=self.dij)
        
        assert isinstance(gof_result, dict)
        assert len(gof_result) == len(exponents)
        for exp in exponents:
            assert exp in gof_result
            assert isinstance(gof_result[exp], pd.DataFrame)
            assert len(gof_result[exp]) == 2  # 2 replications
    
    def test_run_law_random_seed(self):
        """Test reproducibility with random seed for run_law"""
        result1 = tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            random_seed=42
        )
        
        result2 = tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5,
            random_seed=42
        )
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_run_model_random_seed(self):
        """Test reproducibility with random seed for run_model"""
        probabilities = {0.5: tdlm.run_law(
            law='GravExp',
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            exponent=0.5
        )}
        
        result1 = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='UM',
            out_trips=self.Oi,
            repli=1,
            random_seed=42
        )
        
        result2 = tdlm.run_model(
            probabilities=probabilities,
            mass_origin=self.mi,
            mass_destination=self.mj,
            distance=self.dij,
            model='UM',
            out_trips=self.Oi,
            repli=1,
            random_seed=42
        )
    
        np.testing.assert_array_equal(result1, result2)

if __name__ == '__main__':
    pytest.main([__file__])

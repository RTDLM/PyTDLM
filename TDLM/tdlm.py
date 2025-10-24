"""
TDLM: Systematic comparison of trip distribution laws and models in Python

Author: Rémi Perrier (2025)

A Python port of the TDLM R package, with 
numpy-based implementations and parallel processing support for multiple 
exponent values.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Union, Optional, List, Dict
import warnings
from scipy.optimize import minimize_scalar

class _TDLMError(Exception):
    """Custom exception for TDLM errors
    :meta private:
    """
    pass

def _vprint(message, verbose):
    if verbose:
        print(message)

def extract_opportunities(
    mass_destination: np.ndarray,
    distance: np.ndarray,
    processes: Optional[int] = None,
    verbose: bool = True
) -> np.ndarray:
    r"""
    Compute the opportunities matrix $`S_{i,j}`$ Number of opportunities located in a circle 
    of radius $`d_{i,j}`$ centered in $`i`$ (excluding the source and the destination).
    
    Parameters
    ----------
    mass_destination : np.ndarray
        Number of inhabitants at destination $`m_j`$
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    np.ndarray
        Opportunities matrix $`S_{i,j}`$ (n x n)
    """
    n = len(mass_destination)
    
    # Validate inputs
    if distance.shape != (n, n):
        raise _TDLMError(f"distance matrix must be {n}x{n}")
    
    _vprint(f"Computing opportunities matrix for {n} regions...", verbose)
    
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    _vprint(f'Using {num_processes} parallel processes', verbose)
    
    # Prepare arguments for parallel processing
    args_list = [(i, distance[i,:], mass_destination, n) for i in range(n)]
    
    # Use multiprocessing to compute S matrix rows in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(_process_opportunities_row, args_list), 
                           total=n, desc="Computing opportunities", disable=not verbose))
    
    # Collect results into S matrix
    S = np.zeros((n, n))
    for i, row_S in results:
        S[i, :] = row_S
    
    _vprint("Done\n", verbose)
    return S


def _process_opportunities_row(args):
    """Process a single row of the opportunities matrix S with complete vectorization."""
    i, distances_i, mj, n = args
    
    # Initialize row
    row_S = np.zeros(n)
    
    # Create 2D arrays for the j and l dimensions (n×n)
    j_indices = np.arange(n).reshape(n, 1)  # Column vector
    l_indices = np.arange(n).reshape(1, n)  # Row vector
    
    # This creates a matrix of distances from i to l
    distances_il = np.broadcast_to(distances_i, (n, n))
    
    # This creates a column vector of distances from i to j
    distances_ij = distances_i.reshape(n, 1)
    
    # Create masks for all combinations of j and l at once
    # distance condition: dist(i,l) <= dist(i,j)
    distance_mask = distances_il <= distances_ij
    
    # l != i mask
    l_not_i_mask = l_indices != i
    
    # l != j mask
    l_not_j_mask = l_indices != j_indices
    
    # Combine all masks
    combined_mask = distance_mask & l_not_i_mask & l_not_j_mask
    
    # Apply the mask to mj and sum for each j
    # Need to reshape mj for broadcasting
    mj_expanded = mj.reshape(1, n)
    row_S = np.sum(combined_mask * mj_expanded, axis=1)
    
    # Set diagonal to 0 (where i==j)
    row_S[i] = 0
    
    return i, row_S

def run_optimization(
    mass_origin: np.ndarray,
    mass_destination: np.ndarray, 
    distance: np.ndarray,
    obs: np.ndarray,
    opportunity: Optional[np.ndarray] = None,
    law: Union[str, List[str]] = "all",
    model: Union[str, List[str]] = "all",
    out_trips: Optional[np.ndarray] = None,
    in_trips: Optional[np.ndarray] = None,
    average: bool = False,
    repli: int = 1,
    measure: str = "CPC",
    processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    r"""
    Run trip distribution law and model simulations, compute the goodness-of-fit measures, and determine the optimal exponent with respect to the specified measure.
    
    Parameters
    ----------
    mass_origin : np.ndarray
        Number of inhabitants at origin $`m_i`$
    mass_destination : np.ndarray  
        Number of inhabitants at destination $`m_j`$
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    obs : np.ndarray
        Observed trip matrix $`T_{i,j}`$ (n x n)
    opportunity : np.ndarray, optional
        Matrix of opportunities $`S_{i,j}`$ (n x n). Required for "Rad", "RadExt", "Schneider".
        If not provided and required, will be computed automatically.
    law : str or List[str], default "all"
        Trip distribution law. "all" or subset of: 
        ["GravExp", "NGravExp", "GravPow", "NGravPow", "Schneider", "Rad", "RadExt", "Rand"]
    model : str or List[str], default "all"
        Distribution model. "all" or subset of:
        ["UM", "PCM", "ACM", "DCM"]
    out_trips : np.ndarray, optional
        Number of out-commuters $`O_i`$. Required for constrained models.
    in_trips : np.ndarray, optional
        Number of in-commuters $`D_j`$. Required for ACM and DCM models.
    average : bool, default False
        Whether the average mobility flow matrix should be generated instead of the `repli` matrices based on random draws.
    repli : int, default 1
        Number of replications. Note that `repli = 1` if `average = True`.
    measure: str, default "CPC"
        Good-of-fit metric with respect to which to determine the optimal exponent
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    pd.DataFrame
        For each combination of law and model, resulting optimal exponent and corresponding goodness-of-fit measures.
    """
    if average:
        repli = 1

    n = len(mass_origin)
    
    #Check laws
    all_laws = ["GravExp", "NGravExp", "GravPow", "NGravPow", "Schneider", "Rad", "RadExt", "Rand"]
    laws_requiring_opportunities = ["Rad", "RadExt", "Schneider"]
    
    if law == "all":
        selected_laws = all_laws
    else:
        selected_laws = law if isinstance(law, list) else [law]
        invalid = set(selected_laws) - set(all_laws)
        if invalid:
            raise _TDLMError(f"Invalid laws: {invalid}. Available: {['all']+all_laws}")
    
    require_opportunity = set(selected_laws).intersection(set(laws_requiring_opportunities))
    if require_opportunity:
        if opportunity is None:
            _vprint(f"Laws {require_opportunity} require opportunities matrix. Computing automatically...", verbose)
            opportunity = extract_opportunities(mass_destination, distance, processes, verbose)
        if opportunity.shape != (n, n):
            raise _TDLMError(f"opportunities matrix must be {n}x{n}")
            
    #Check models
    all_models = ["UM", "PCM", "ACM", "DCM"]
    if model == "all":
        selected_models = all_models
    else:
        selected_models = model if isinstance(model, list) else [model]
        invalid = set(selected_models) - set(all_models)
        if invalid:
            raise _TDLMError(f"Invalid models: {invalid}. Available: {['all']+all_models}")
    
    # Check array dimensions
    if len(mass_destination) != n:
        raise _TDLMError("mass_origin and mass_destination must have same length")
    
    if distance.shape != (n, n):
        raise _TDLMError(f"distance matrix must be {n}x{n}")
    
    # Check trip constraints for models
    if set(selected_models).intersection(set(["PCM", "DCM"])) and out_trips is None:
        raise _TDLMError(f"out_trips required for model '{model}'")
    
    if set(selected_models).intersection(set(["ACM", "DCM"])) and in_trips is None:
        raise _TDLMError(f"in_trips required for model '{model}'")
    
    if out_trips is not None and len(out_trips) != n:
        raise _TDLMError("out_trips must have same length as mass arrays")
        
    if in_trips is not None and len(in_trips) != n:
        raise _TDLMError("in_trips must have same length as mass arrays")
    
    #Check measure
    all_measures = ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    if measure not in all_measures:
        raise _TDLMError(f"Invalid measure: {measure}. Available: {all_measures}")
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    results = []
    for i, l in enumerate(selected_laws):
        for j, m in enumerate(selected_models):
            result_dict= {'Law':l, 'Model':m}
            _vprint(f'Calculating average {measure} for {l} with {m} over {repli} realizations', verbose)
            params = [l, m, measure, mass_origin, mass_destination, distance, opportunity, out_trips, in_trips, obs, average, repli, processes, verbose]
            res = minimize_scalar(_single_metric_average, args=(params))
            if res.success:
                result_dict['Exponent'] = res.x
            else:
                result_dict['Exponent'] = np.nan
            results.append(result_dict)
    
    return pd.DataFrame(results)

def run_law_model_gof(
    law: str,
    mass_origin: np.ndarray,
    mass_destination: np.ndarray, 
    distance: np.ndarray,
    obs: np.ndarray,
    opportunity: Optional[np.ndarray] = None,
    exponent: Union[float, np.ndarray] = 1.0,
    model: str = "UM",
    out_trips: Optional[np.ndarray] = None,
    in_trips: Optional[np.ndarray] = None,
    average: bool = False,
    repli: int = 1,
    measures: Union[str, List[str]] = "all",
    processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Union[pd.DataFrame, Dict[float, pd.DataFrame]]:
    r"""
    Run trip distribution law and model simulations, and compute the goodness-of-fit measures, all in one go.
    
    Parameters
    ----------
    law : str
        Trip distribution law. One of: "GravExp", "NGravExp", "GravPow", 
        "NGravPow", "Schneider", "Rad", "RadExt", "Rand"
    mass_origin : np.ndarray
        Number of inhabitants at origin $`m_i`$
    mass_destination : np.ndarray  
        Number of inhabitants at destination $`m_j`$
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    obs : np.ndarray
        Observed trip matrix $`T_{i,j}`$ (n x n)
    opportunity : np.ndarray, optional
        Matrix of opportunities $`S_{i,j}`$ (n x n). Required for "Rad", "RadExt", "Schneider".
        If not provided and required, will be computed automatically.
    exponent : float or np.ndarray
        Exponent parameter(s) for the distribution law
    model : str, default "UM"
        Distribution model. One of: "UM", "PCM", "ACM", "DCM"
    out_trips : np.ndarray, optional
        Number of out-commuters $`O_i`$. Required for constrained models.
    in_trips : np.ndarray, optional
        Number of in-commuters $`D_j`$. Required for ACM and DCM models.
    average : bool, default False
        Whether the average mobility flow matrix should be generated instead of the `repli` matrices based on random draws.
    repli : int, default 1
        Number of replications. Note that `repli = 1` if `average = True`.
    measures : str or List[str], default "all"
        Measures to calculate. "all" or subset of:
        ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    Union[pd.DataFrame, Dict[float, pd.DataFrame]]
        If single exponent: DataFrame with measures.
        If multiple exponents: Dict with exponents as keys, DataFrames as values.
    """
    if average:
        repli = 1
    pij = None
    
    # Check if opportunities matrix is needed and compute if not provided
    laws_requiring_opportunities = ["Rad", "RadExt", "Schneider"]
    if law in laws_requiring_opportunities and opportunity is None:
        _vprint(f"Law '{law}' requires opportunities matrix. Computing automatically...", verbose)
        opportunity = extract_opportunities(mass_destination, distance, processes, verbose)
    
    # Input validation
    _validate_inputs(law, model, mass_origin, mass_destination, distance, 
                    opportunity, out_trips, in_trips)
    # Available measures
    all_measures = ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    
    if measures == "all":
        selected_measures = all_measures
    else:
        selected_measures = measures if isinstance(measures, list) else [measures]
        invalid = set(selected_measures) - set(all_measures)
        if invalid:
            raise _TDLMError(f"Invalid measures: {invalid}. Available: {['all']+all_measures}")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Handle single vs multiple exponents
    exponents = np.atleast_1d(exponent)
    single_exponent = len(exponents) == 1
    
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    if len(exponents) > 1 and num_processes > 1:
        # Parallel processing for multiple exponents
        _vprint(f'Running simulations and GOF for {law} with {model} model ({repli} replications)', verbose)
        _vprint(f'Using {num_processes} parallel processes', verbose)
        
        shm_list = [] # To track shared memory blocks for cleanup
        try:
            # Create shared memory for all large NumPy arrays
            shm_info = {
                'mass_origin': _numpy_to_shm(mass_origin, shm_list),
                'mass_destination': _numpy_to_shm(mass_destination, shm_list),
                'distance': _numpy_to_shm(distance, shm_list),
                'opportunity': _numpy_to_shm(opportunity, shm_list),
                'out_trips': _numpy_to_shm(out_trips, shm_list),
                'in_trips': _numpy_to_shm(in_trips, shm_list),
                'obs': _numpy_to_shm(obs, shm_list)
            }
            
            with mp.Pool(processes=num_processes) as pool:
                params = [(shm_info, pij, law, model, beta, repli, average, selected_measures) for beta in exponents]
                results = list(tqdm(pool.imap(_process_exponent_gof_shared, params),
                                    total=len(exponents), desc='Computing exponents', disable=not verbose))
        finally:
            # CRITICAL: Clean up! Unlink all shared memory blocks
            # This must be in a 'finally' block to run even if the pool fails
            _vprint("Cleaning up shared memory blocks...", verbose)
            for shm in shm_list:
                shm.close()
                shm.unlink() # Destroys the shared memory block
                
        # Organize results
        output = {}
        for i, exponent in enumerate(exponents):
            output[exponent] = results[i]
        _vprint('Done\n', verbose)
        return output
    else:
        # Sequential processing
        data = (mass_origin, mass_destination, distance, opportunity, out_trips, in_trips, obs)
        if single_exponent:
            beta = exponents[0]
            _vprint(f'Simulating matrix and GOF for {law} β = {beta:.2g} with {model}', verbose)
            params = (data, pij, law, model, beta, repli, average, selected_measures)
            result = _process_exponent_gof(params)
            return result
        else:
            results = {}
            _vprint(f'Running simulations and GOF for {law} with {model} model ({repli} replications)', verbose)
            for i, beta in enumerate(tqdm(exponents, desc='Computing exponents', disable=not verbose)):
                params = (data, pij, law, model, beta, repli, average, selected_measures)
                results[beta] = _process_exponent_gof(params)
            return results
        
def run_law_model(
    law: str,
    mass_origin: np.ndarray,
    mass_destination: np.ndarray, 
    distance: np.ndarray,
    opportunity: Optional[np.ndarray] = None,
    exponent: Union[float, np.ndarray] = 1.0,
    return_proba: bool = False,
    model: str = "UM",
    out_trips: Optional[np.ndarray] = None,
    in_trips: Optional[np.ndarray] = None,
    average: bool = False,
    repli: int = 1,
    processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True    
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    r"""
    Run trip distribution law and model simulations.
    
    Parameters
    ----------
    law : str
        Trip distribution law. One of: "GravExp", "NGravExp", "GravPow", 
        "NGravPow", "Schneider", "Rad", "RadExt", "Rand"
    mass_origin : np.ndarray
        Number of inhabitants at origin $`m_i`$
    mass_destination : np.ndarray  
        Number of inhabitants at destination $`m_j`$
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    opportunity : np.ndarray, optional
        Matrix of opportunities $`S_{i,j}`$ (n x n). Required for "Rad", "RadExt", "Schneider".
        If not provided and required, will be computed automatically.
    exponent : float or np.ndarray
        Exponent parameter(s) for the distribution law
    return_proba : bool, default False
        Whether to return probability matrices
    model : str, default "UM"
        Distribution model. One of: "UM", "PCM", "ACM", "DCM"
    out_trips : np.ndarray, optional
        Number of out-commuters $`O_i`$. Required for constrained models.
    in_trips : np.ndarray, optional
        Number of in-commuters $`D_j`$. Required for ACM and DCM models.
    average : bool, default False
        Whether the average mobility flow matrix should be generated instead of the `repli` matrices based on random draws.
    repli : int, default 1
        Number of replications. Note that `repli = 1` if `average = True`.
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    Union[np.ndarray, Dict[float, np.ndarray]]
        Simulated trip matrix or matrices $`\tilde{T}_{i,j}`$.
        If single exponent: np.ndarray of shape (repli, n, n).
        If multiple exponents: Dict with exponents as keys, arrays as values.
    """
    if average:
        repli = 1
    pij = None
    
    # Check if opportunities matrix is needed and compute if not provided
    laws_requiring_opportunities = ["Rad", "RadExt", "Schneider"]
    if law in laws_requiring_opportunities and opportunity is None:
        _vprint(f"Law '{law}' requires opportunities matrix. Computing automatically...", verbose)
        opportunity = extract_opportunities(mass_destination, distance, processes, verbose)
    
    # Input validation
    _validate_inputs(law, model, mass_origin, mass_destination, distance, 
                    opportunity, out_trips, in_trips)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Handle single vs multiple exponents
    exponents = np.atleast_1d(exponent)
    single_exponent = len(exponents) == 1
    
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    if len(exponents) > 1 and num_processes > 1:
        _vprint(f'Running simulations for {law} with {model} model ({repli} replications)', verbose)
        _vprint(f'Using {num_processes} parallel processes', verbose)
        
        shm_list = [] # To track shared memory blocks for cleanup
        try:
            # Create shared memory for all large NumPy arrays
            shm_info = {
                'mass_origin': _numpy_to_shm(mass_origin, shm_list),
                'mass_destination': _numpy_to_shm(mass_destination, shm_list),
                'distance': _numpy_to_shm(distance, shm_list),
                'opportunity': _numpy_to_shm(opportunity, shm_list),
                'out_trips': _numpy_to_shm(out_trips, shm_list),
                'in_trips': _numpy_to_shm(in_trips, shm_list),
            }

            with mp.Pool(processes=num_processes) as pool:
                # Prepare parameters for the shared worker function
                # Pass the dict of shared memory info instead of the raw data
                params = [(shm_info, pij, law, model, beta, repli, return_proba, average) for beta in exponents]
                
                results = list(tqdm(pool.imap(_process_exponent_shared, params),
                                      total=len(exponents), desc='Computing exponents', disable=not verbose))
        finally:
            # CRITICAL: Clean up! Unlink all shared memory blocks
            # This must be in a 'finally' block to run even if the pool fails
            _vprint("Cleaning up shared memory blocks...", verbose)
            for shm in shm_list:
                shm.close()
                shm.unlink() # Destroys the shared memory block

        # Organize results
        output = {}
        for i, beta in enumerate(exponents):
            if return_proba:
                output[beta] = results[i]
            else:
                output[beta] = results[i]['simulations']
    
    else:
        # Sequential processing
        n = len(mass_origin)
        data = (n, mass_origin, mass_destination, out_trips, in_trips, distance, opportunity)

        output = {}
        if single_exponent:
            beta = exponents[0]
            _vprint(f'Simulating matrix for {law} β = {beta:.2g} with {model}', verbose)
            params = (data, pij, law, model, beta, repli, return_proba, average)
            result = _process_exponent(params) # Original function call
            if return_proba:
                output[beta] = result
            else:
                output[beta] = result['simulations']
        else:
            _vprint(f'Running simulations for {law} with {model} model ({repli} replications)', verbose)
            for i, beta in enumerate(tqdm(exponents, desc='Computing exponents', disable=not verbose)):
                params = (data, pij, law, model, beta, repli, return_proba, average)
                result = _process_exponent(params) # Original function call
                if return_proba:
                    output[beta] = result
                else:
                    output[beta] = result['simulations']

    _vprint('Done\n', verbose)
    if single_exponent:
        return list(output.values())[0]
    else:
        return output


def run_law(
    law: str,
    mass_origin: np.ndarray,
    mass_destination: np.ndarray, 
    distance: np.ndarray,
    opportunity: Optional[np.ndarray] = None,
    exponent: Union[float, np.ndarray] = 1.0,
    processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    r"""
    Estimate the probability matrix $`p_{i,j}`$ according to the trip distribution law.
    
    Parameters
    ----------
    law : str
        Trip distribution law. One of: "GravExp", "NGravExp", "GravPow", 
        "NGravPow", "Schneider", "Rad", "RadExt", "Rand"
    mass_origin : np.ndarray
        Number of inhabitants at origin $`m_i`$
    mass_destination : np.ndarray  
        Number of inhabitants at destination $`m_j`$
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    opportunity : np.ndarray, optional
        Matrix of opportunities $`S_{i,j}`$ (n x n). Required for "Rad", "RadExt", "Schneider".
        If not provided and required, will be computed automatically.
    exponent : float or np.ndarray
        Exponent parameter(s) for the distribution law
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    Union[np.ndarray, Dict[float, np.ndarray]]
        Estimated matrix or matrices of probabilities $`p_{i,j}`$.
        If single exponent: np.ndarray of shape (n, n).
        If multiple exponents: Dict with exponents as keys, arrays as values.
    """

    # Check if opportunities matrix is needed and compute if not provided
    laws_requiring_opportunities = ["Rad", "RadExt", "Schneider"]
    if law in laws_requiring_opportunities and opportunity is None:
        _vprint(f"Law '{law}' requires opportunities matrix. Computing automatically...", verbose)
        opportunity = extract_opportunities(mass_destination, distance, processes, verbose)
    
    # Input validation
    _validate_inputs(law, "UM", mass_origin, mass_destination, distance, 
                    opportunity, None, None)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Handle single vs multiple exponents
    exponents = np.atleast_1d(exponent)
    single_exponent = len(exponents) == 1
    
 
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    if len(exponents) > 1 and num_processes > 1:
        # Parallel processing for multiple exponents
        _vprint(f'Estimating probabilities for {law}', verbose)
        _vprint(f'Using {num_processes} parallel processes', verbose)
        
        with mp.Pool(processes=num_processes) as pool:
            params = [(law, distance, opportunity, mass_origin, mass_destination, beta) for beta in exponents]
            results = list(tqdm(pool.starmap(_proba, params), 
                              total=len(exponents), desc='Computing exponents', disable=not verbose))
        
        # Organize results
        output = {}
        for i, beta in enumerate(exponents):
            output[beta] = results[i]
                
    else:
        # Sequential processing
        output = {}
        if single_exponent:
            beta = exponents[0]
            _vprint(f'Estimating probabilities for {law} with β = {beta:.2g}', verbose)
            return _proba(law, distance, opportunity, mass_origin, mass_destination, beta)

        else:
            _vprint(f'Estimating probabilities for {law}', verbose)
            
            for i, beta in enumerate(tqdm(exponents, desc='Computing exponents', disable=not verbose)):
                output[beta] = _proba(law, distance, opportunity, mass_origin, mass_destination, beta)
    _vprint('Done\n', verbose)
    
    return output

def run_model(
    probabilities: Dict[float, np.ndarray],
    mass_origin: np.ndarray,
    mass_destination: np.ndarray, 
    distance: np.ndarray,
    model: str = "UM",
    out_trips: Optional[np.ndarray] = None,
    in_trips: Optional[np.ndarray] = None,
    average: bool = False,
    repli: int = 1,
    processes: Optional[int] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    r"""
    Run trip distribution model simulations with provided probability matrix or matrices $`p_{i,j}`$.
    
    Parameters
    ----------
    probabilities : Dict[float, np.ndarray]
        Estimated matrix or matrices of probabilities $`p_{i,j}`$.
        Dict with exponent(s) as key(s), arrays as values.
    mass_origin : np.ndarray
        Number of inhabitants at origin $`m_i`$
    mass_destination : np.ndarray  
        Number of inhabitants at destination $`m_j`$
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    model : str, default "UM"
        Distribution model. One of: "UM", "PCM", "ACM", "DCM"
    out_trips : np.ndarray, optional
        Number of out-commuters $`O_i`$. Required for constrained models.
    in_trips : np.ndarray, optional
        Number of in-commuters $`D_j`$. Required for ACM and DCM models.
    average : bool, default False
        Whether the average mobility flow matrix should be generated instead of the `repli` matrices based on random draws.
    repli : int, default 1
        Number of replications. Note that `repli = 1` if `average = True`.
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    Union[np.ndarray, Dict[float, np.ndarray]]
        Simulated trip matrix or matrices $`\tilde{T}_{i,j}`$.
        If single exponent: np.ndarray of shape (repli, n, n).
        If multiple exponents: Dict with exponents as keys, arrays as values.
    """
    if average:
        repli = 1
    # Reconstruct exponent or exponents array from provided probabilities
    exponent = np.fromiter(probabilities.keys(), dtype=float)
    exponent.sort()
    
    # Input validation
    _validate_inputs("Rand", model, mass_origin, mass_destination, distance, 
                    None, out_trips, in_trips)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Handle single vs multiple exponents
    exponents = np.atleast_1d(exponent)
    single_exponent = len(exponents) == 1
    
    opportunity = None
    return_proba = False
    law = None
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    if len(exponents) > 1 and num_processes > 1:
        # Parallel processing for multiple exponents
        _vprint(f'Running simulations with {model} model ({repli} replications)', verbose)
        _vprint(f'Using {num_processes} parallel processes', verbose)
        shm_list = [] # To track shared memory blocks for cleanup
        try:
            # Create shared memory for all large NumPy arrays
            shm_info = {
                'mass_origin': _numpy_to_shm(mass_origin, shm_list),
                'mass_destination': _numpy_to_shm(mass_destination, shm_list),
                'distance': _numpy_to_shm(distance, shm_list),
                'opportunity': _numpy_to_shm(opportunity, shm_list),
                'out_trips': _numpy_to_shm(out_trips, shm_list),
                'in_trips': _numpy_to_shm(in_trips, shm_list),
            }
            
            with mp.Pool(processes=num_processes) as pool:
                # Prepare parameters for the shared worker function
                # Pass the dict of shared memory info instead of the raw data
                params = [(shm_info, probabilities[beta], law, model, beta, repli, return_proba, average) for beta in exponents]
                
                results = list(tqdm(pool.imap(_process_exponent_shared, params),
                                      total=len(exponents), desc='Computing exponents', disable=not verbose))
             
        finally:
            # CRITICAL: Clean up! Unlink all shared memory blocks
            # This must be in a 'finally' block to run even if the pool fails
            _vprint("Cleaning up shared memory blocks...", verbose)
            for shm in shm_list:
                shm.close()
                shm.unlink() # Destroys the shared memory block
        # Organize results
        output = {}
        for i, beta in enumerate(exponents):
            output[beta] = results[i]['simulations']
            
    else:
        # Sequential processing
        n = len(mass_origin)
        data = (n, mass_origin, mass_destination, out_trips, in_trips, distance, opportunity)

        output = {}
        if single_exponent:
            beta = exponents[0]
            _vprint(f'Simulating matrix with {model}', verbose)
            params = (data, probabilities[beta], law, model, beta, repli, return_proba, average)
            result = _process_exponent(params)
            output[beta] = result['simulations']
        else:
            _vprint(f'Running simulations with {model} model ({repli} replications)', verbose)
            for i, beta in enumerate(tqdm(exponents, desc='Computing exponents', disable=not verbose)):
                params = (data, probabilities[beta], law, model, beta, repli, return_proba, average)
                result = _process_exponent(params)
                output[beta] = result['simulations']
    _vprint('Done\n', verbose)
    
    # Return format based on input
    if single_exponent:
        return list(output.values())[0]
    else:
        return output
    
def gof(
    sim: Union[np.ndarray, Dict[float, np.ndarray]], 
    obs: np.ndarray,
    distance: np.ndarray,
    measures: Union[str, List[str]] = "all",
    processes: Optional[int] = None,
    verbose: bool = True
) -> Union[pd.DataFrame, Dict[float, pd.DataFrame]]:
    r"""
    Calculate goodness-of-fit measures for simulated vs observed trip matrices.
    
    Parameters
    ----------
    sim : np.ndarray or Dict[float, np.ndarray]
        Simulated trip matrix or matrices $`\tilde{T}_{i,j}`$. If Dict, keys should be exponent values
    obs : np.ndarray
        Observed trip matrix $`T_{i,j}`$ (n x n)
    distance : np.ndarray
        Distance matrix $`d_{i,j}`$ (n x n)
    measures : str or List[str], default "all"
        Measures to calculate. "all" or subset of:
        ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    verbose : bool, default True
        Controls the verbosity
        
    Returns
    -------
    Union[pd.DataFrame, Dict[float, pd.DataFrame]]
        If single exponent: DataFrame with measures.
        If multiple exponents: Dict with exponents as keys, DataFrames as values.
    """
    
    # Available measures
    all_measures = ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    
    if measures == "all":
        selected_measures = all_measures
    else:
        selected_measures = measures if isinstance(measures, list) else [measures]
        invalid = set(selected_measures) - set(all_measures)
        if invalid:
            raise _TDLMError(f"Invalid measures: {invalid}. Available: {['all']+all_measures}")
    
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    # Handle single vs multiple simulations
    if isinstance(sim, dict):
        exponents = list(sim.keys())
        single_simulation = len(exponents) == 1
        
        if len(exponents) > 1 and num_processes > 1:
            # Parallel processing for multiple exponents
            _vprint(f'Calculating GOF measures for {len(exponents)} exponents', verbose)
            _vprint(f'Using {num_processes} parallel processes', verbose)
            
            shm_list = [] # To track shared memory blocks for cleanup
            try:
                # Create shared memory for all large NumPy arrays
                shm_info = {
                    'obs': _numpy_to_shm(obs, shm_list),
                    'distance': _numpy_to_shm(distance, shm_list),
                }
                
                with mp.Pool(processes=num_processes) as pool:
                    # Prepare parameters for the new shared worker function
                    # Pass the dict of shared memory info instead of the raw data
                    params = [(shm_info, sim[exponent], selected_measures) for exponent in exponents]
                    
                    results = list(tqdm(pool.imap(_calculate_gof_shared, params), 
                                  total=len(exponents), desc='Computing GOF measures', disable=not verbose))
            finally:
                # CRITICAL: Clean up! Unlink all shared memory blocks
                # This must be in a 'finally' block to run even if the pool fails
                _vprint("Cleaning up shared memory blocks...", verbose)
                for shm in shm_list:
                    shm.close()
                    shm.unlink() # Destroys the shared memory block
                    
            # Organize results
            output = {}
            for i, exponent in enumerate(exponents):
                output[exponent] = results[i]
            
            _vprint('Done\n', verbose)
            return output
        
        else:
            # Sequential processing
            results = {}
            if single_simulation:
                exponent = exponents[0]
                _vprint(f'Calculating GOF measures for exponent {exponent}', verbose)
                params = (sim[exponent], obs, distance, selected_measures)
                results[exponent] = _calculate_gof(params)
            else:
                _vprint(f'Calculating GOF measures for {len(exponents)} exponents', verbose)
                for exponent in tqdm(exponents, desc='Computing GOF measures', disable=not verbose):
                    params = (sim[exponent], obs, distance, selected_measures)
                    results[exponent] = _calculate_gof(params)
            
            _vprint('Done\n', verbose)
            return results
    else:
        # Single simulation matrix
        _vprint('Calculating GOF measures', verbose)
        result = _calculate_gof((sim, obs, distance, selected_measures))
        _vprint('Done\n', verbose)
        return result

def _validate_inputs(law, model, mass_origin, mass_destination, distance, 
                    opportunity, out_trips, in_trips):
    """Validate input parameters"""
    
    valid_laws = ["GravExp", "NGravExp", "GravPow", "NGravPow", "Schneider", "Rad", "RadExt", "Rand"]
    valid_models = ["UM", "PCM", "ACM", "DCM"]
    
    if law not in valid_laws:
        raise _TDLMError(f"Invalid law '{law}'. Must be one of: {valid_laws}")
    
    if model not in valid_models:
        raise _TDLMError(f"Invalid model '{model}'. Must be one of: {valid_models}")
    
    # Check array dimensions
    n = len(mass_origin)
    if len(mass_destination) != n:
        raise _TDLMError("mass_origin and mass_destination must have same length")
    
    if distance.shape != (n, n):
        raise _TDLMError(f"distance matrix must be {n}x{n}")
    
    # Check opportunities matrix for relevant laws
    if law in ["Rad", "RadExt", "Schneider"]:
        if opportunity is None:
            raise _TDLMError(f"opportunities matrix required for law '{law}'")
        if opportunity.shape != (n, n):
            raise _TDLMError(f"opportunities matrix must be {n}x{n}")
    
    # Check trip constraints for models
    if model in ["PCM", "DCM"] and out_trips is None:
        raise _TDLMError(f"out_trips required for model '{model}'")
    
    if model in ["ACM", "DCM"] and in_trips is None:
        raise _TDLMError(f"in_trips required for model '{model}'")
    
    if out_trips is not None and len(out_trips) != n:
        raise _TDLMError("out_trips must have same length as mass arrays")
        
    if in_trips is not None and len(in_trips) != n:
        raise _TDLMError("in_trips must have same length as mass arrays")

def _numpy_to_shm(arr, shm_list):
    """Copies a NumPy array into a new shared memory block."""
    if arr is None:
        return None
    # Create a shared memory block with the exact size of the array
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    # Create a NumPy array that uses the shared memory buffer
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    # Copy the data from the original array to the shared array
    shm_arr[:] = arr[:]
    # Keep track of the shm object for cleanup in the main process
    shm_list.append(shm)
    # Return the metadata needed to reconstruct the array in the worker
    return (shm.name, arr.shape, arr.dtype)

def _shm_to_numpy(shm_meta):
    """Reconstructs a NumPy array from shared memory metadata in a worker."""
    if shm_meta is None:
        return None, None
    shm_name, shape, dtype = shm_meta
    # Attach to the existing shared memory block by name
    shm = shared_memory.SharedMemory(name=shm_name)
    # Create a NumPy array that points to the buffer, no data is copied
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return arr, shm

def _process_exponent_shared(param):
    """
    Worker process for a single exponent that uses data from shared memory.
    """
    shm_info, pij, law, model, beta, repli, return_proba, average = param
    attached_shms = []
    try:
        # Reconstruct all necessary arrays from shared memory metadata
        mass_origin, shm_mo = _shm_to_numpy(shm_info['mass_origin'])
        mass_destination, shm_md = _shm_to_numpy(shm_info['mass_destination'])
        distance, shm_d = _shm_to_numpy(shm_info['distance'])
        opportunity, shm_o = _shm_to_numpy(shm_info['opportunity'])
        out_trips, shm_ot = _shm_to_numpy(shm_info['out_trips'])
        in_trips, shm_it = _shm_to_numpy(shm_info['in_trips'])

        # Keep track of all attached blocks to ensure they are closed
        attached_shms = [s for s in [shm_mo, shm_md, shm_d, shm_o, shm_ot, shm_it] if s is not None]

        n = len(mass_origin)
        
        if pij is None:
            # Build the matrix pij according to the law
            pij = _proba(law, distance, opportunity, mass_origin, mass_destination, beta)
            
        simulations = []
        for r in range(repli):
            S = np.zeros((n, n))
            if model == "UM":
                S = _UM(pij, out_trips, average)
            elif model == "PCM":
                S = _PCM(pij, out_trips, average)
            elif model == "ACM":
                S = _ACM(pij, in_trips, average)
            elif model == "DCM":
                S = _DCM(pij, out_trips, in_trips, 50, 0.01, average)
            simulations.append(S)

        simulations = np.array(simulations)

        if return_proba:
            sumpij = np.sum(pij)
            return {
                'simulations': simulations,
                'probabilities': pij / sumpij if sumpij > 0 else pij
            }
        else:
            return {'simulations': simulations}

    finally:
        # CRITICAL: Each worker must close its connection to the shared memory blocks
        for shm in attached_shms:
            shm.close()

def _process_exponent(params):
    """Process a single exponent value"""
    data, pij, law, model, beta, repli, return_proba, average = params
    n, mi, mj, Oi, Dj, dij, sij = data
    
    # Build the matrix pij according to the law
    if pij is None:
        pij = _proba(law, dij, sij, mi, mj, beta)
    
    # Store results
    simulations = []
    
    # Loop replications
    for r in range(repli):
        # Simulated OD
        S = np.zeros((n, n))

        # Network generation according to the constrained model
        if model == "UM":  # Unconstrained model
            S = _UM(pij, Oi, average)
        elif model == "PCM":  # Production constrained model
            S = _PCM(pij, Oi, average)
        elif model == "ACM":  # Attraction constrained model
            S = _ACM(pij, Dj, average)
        elif model == "DCM":  # Doubly constrained model
            S = _DCM(pij, Oi, Dj, 50, 0.01, average)

        simulations.append(S)
    
    simulations = np.array(simulations)
    
    if return_proba:
        # Normalize pij
        sumpij = np.sum(pij)
        return {
            'simulations': simulations,
            'probabilities': pij / sumpij if sumpij > 0 else pij
        }
    else:
        return {'simulations': simulations}

def _process_exponent_gof_shared(params):
    """Run simulations and process GOF calculation for a single exponent"""
    shm_info, pij, law, model, beta, repli, average, measures = params
    attached_shms = []
    try:
        # Reconstruct all necessary arrays from shared memory metadata
        mi, shm_mo = _shm_to_numpy(shm_info['mass_origin'])
        mj, shm_md = _shm_to_numpy(shm_info['mass_destination'])
        dij, shm_d = _shm_to_numpy(shm_info['distance'])
        sij, shm_o = _shm_to_numpy(shm_info['opportunity'])
        Oi, shm_ot = _shm_to_numpy(shm_info['out_trips'])
        Dj, shm_it = _shm_to_numpy(shm_info['in_trips'])
        Tij, shm_ob = _shm_to_numpy(shm_info['obs'])
        
        # Keep track of all attached blocks to ensure they are closed
        attached_shms = [s for s in [shm_mo, shm_md, shm_d, shm_o, shm_ot, shm_it, shm_ob] if s is not None]

        n = len(mi)
        
        if pij is None:
            pij = _proba(law, dij, sij, mi, mj, beta)
        
        # Prepare observed data
        pobs = (Tij / Tij.sum()).flatten()
        nb = np.sum(Tij)
        T_range = np.max(Tij) - np.min(Tij)
        
        # Calculate distance indices for CPCd
        indices = np.floor(dij / 2).astype(int).flatten()
        max_index = indices.max() + 1
        CDD_R = np.bincount(indices, weights=Tij.flatten(), minlength=max_index)
        
        results = []
        for r in range(repli):
            # Simulated OD
            S = np.zeros((n, n))
    
            # Network generation according to the constrained model
            if model == "UM":  # Unconstrained model
                S = _UM(pij, Oi, average)
            elif model == "PCM":  # Production constrained model
                S = _PCM(pij, Oi, average)
            elif model == "ACM":  # Attraction constrained model
                S = _ACM(pij, Dj, average)
            elif model == "DCM":  # Doubly constrained model
                S = _DCM(pij, Oi, Dj, 50, 0.01, average)
            
            result_dict = {"Replication": r}
            
            if "CPC" in measures:
                # CPC - Common Part of Commuters
                mask = (Tij != 0) * (S != 0)
                cpc = np.minimum(Tij[mask], S[mask]).sum() / nb if nb > 0 else 0
                result_dict["CPC"] = cpc
            
            if "CPL" in measures:
                # CPL - Common Part of Links
                nbNL = ((Tij == 0) * (S != 0)).sum()  # Number of new links
                nbML = ((Tij != 0) * (S == 0)).sum()  # Number of missing links
                nbCL = ((Tij != 0) * (S != 0)).sum()  # Number of common links
                cpl = 2 * nbCL / (nbNL + 2 * nbCL + nbML) if (nbNL + 2 * nbCL + nbML) > 0 else 0
                result_dict["CPL"] = cpl
            
            if "CPCd" in measures:
                # CPCd - Common Part of Commuters by distance
                CDD_S = np.bincount(indices, weights=S.flatten(), minlength=max_index)
                cpcd = (np.abs(CDD_S - CDD_R) / nb).sum() if nb > 0 else 0
                cpcd = 1 - 0.5 * cpcd
                result_dict["CPCd"] = cpcd
            
            if "KS_stat" in measures or "KS_pval" in measures:
                # KS - Kolmogorov-Smirnov test
                ks_statistic, ks_pvalue = _ks_weighted(
                    data1=dij.flatten(), 
                    wei1=Tij.flatten(), 
                    wei2=S.flatten()
                )
                if "KS_stat" in measures:
                    result_dict["KS_stat"] = ks_statistic
                if "KS_pval" in measures:
                    result_dict["KS_pval"] = ks_pvalue
            
            if "KL_div" in measures:
                # KL - Kullback-Leibler divergence
                ppred = (S / S.sum()).flatten() if S.sum() > 0 else S.flatten()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kl_div_array = pobs * np.log(pobs / ppred)
                    kl_div = np.nan_to_num(kl_div_array, nan=0., posinf=0., neginf=0.).sum()
                result_dict["KL_div"] = kl_div
            
            if "RMSE" in measures:
                # NRMSE - Normalized Root Mean Square Error
                if T_range > 0 and Tij.sum() > 0:
                    mse = np.sum((Tij - S) ** 2) / Tij.sum()
                    nrmse = np.sqrt(mse)
                else:
                    nrmse = 0
                result_dict["RMSE"] = nrmse
            
            results.append(result_dict)
            
        return pd.DataFrame(results)
    finally:
        # CRITICAL: Each worker must close its connection to the shared memory blocks
        for shm in attached_shms:
            shm.close()

def _process_exponent_gof(params):
    """Run simulations and process GOF calculation for a single exponent"""
    data, pij, law, model, beta, repli, average, measures = params
    mi, mj, dij, sij, Oi, Dj, Tij = data
    n = len(mi)
    
    if pij is None:
        pij = _proba(law, dij, sij, mi, mj, beta)
    
    # Prepare observed data
    pobs = (Tij / Tij.sum()).flatten()
    nb = np.sum(Tij)
    T_range = np.max(Tij) - np.min(Tij)
    
    # Calculate distance indices for CPCd
    indices = np.floor(dij / 2).astype(int).flatten()
    max_index = indices.max() + 1
    CDD_R = np.bincount(indices, weights=Tij.flatten(), minlength=max_index)
    
    results = []
    for r in range(repli):
        # Simulated OD
        S = np.zeros((n, n))

        # Network generation according to the constrained model
        if model == "UM":  # Unconstrained model
            S = _UM(pij, Oi, average)
        elif model == "PCM":  # Production constrained model
            S = _PCM(pij, Oi, average)
        elif model == "ACM":  # Attraction constrained model
            S = _ACM(pij, Dj, average)
        elif model == "DCM":  # Doubly constrained model
            S = _DCM(pij, Oi, Dj, 50, 0.01, average)
        
        result_dict = {"Replication": r}
        
        if "CPC" in measures:
            # CPC - Common Part of Commuters
            mask = (Tij != 0) * (S != 0)
            cpc = np.minimum(Tij[mask], S[mask]).sum() / nb if nb > 0 else 0
            result_dict["CPC"] = cpc
        
        if "CPL" in measures:
            # CPL - Common Part of Links
            nbNL = ((Tij == 0) * (S != 0)).sum()  # Number of new links
            nbML = ((Tij != 0) * (S == 0)).sum()  # Number of missing links
            nbCL = ((Tij != 0) * (S != 0)).sum()  # Number of common links
            cpl = 2 * nbCL / (nbNL + 2 * nbCL + nbML) if (nbNL + 2 * nbCL + nbML) > 0 else 0
            result_dict["CPL"] = cpl
        
        if "CPCd" in measures:
            # CPCd - Common Part of Commuters by distance
            CDD_S = np.bincount(indices, weights=S.flatten(), minlength=max_index)
            cpcd = (np.abs(CDD_S - CDD_R) / nb).sum() if nb > 0 else 0
            cpcd = 1 - 0.5 * cpcd
            result_dict["CPCd"] = cpcd
        
        if "KS_stat" in measures or "KS_pval" in measures:
            # KS - Kolmogorov-Smirnov test
            ks_statistic, ks_pvalue = _ks_weighted(
                data1=dij.flatten(), 
                wei1=Tij.flatten(), 
                wei2=S.flatten()
            )
            if "KS_stat" in measures:
                result_dict["KS_stat"] = ks_statistic
            if "KS_pval" in measures:
                result_dict["KS_pval"] = ks_pvalue
        
        if "KL_div" in measures:
            # KL - Kullback-Leibler divergence
            ppred = (S / S.sum()).flatten() if S.sum() > 0 else S.flatten()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kl_div_array = pobs * np.log(pobs / ppred)
                kl_div = np.nan_to_num(kl_div_array, nan=0., posinf=0., neginf=0.).sum()
            result_dict["KL_div"] = kl_div
        
        if "RMSE" in measures:
            # NRMSE - Normalized Root Mean Square Error
            if T_range > 0 and Tij.sum() > 0:
                mse = np.sum((Tij - S) ** 2) / Tij.sum()
                nrmse = np.sqrt(mse)
            else:
                nrmse = 0
            result_dict["RMSE"] = nrmse
        
        results.append(result_dict)
        
    return pd.DataFrame(results)

def _calculate_gof(params):
    """Calculate goodness-of-fit measures"""
    sim_matrices, obs, distance, measures = params
    # Ensure sim_matrices is 3D (replications, n, n)
    if sim_matrices.ndim == 2:
        sim_matrices = sim_matrices[np.newaxis, ...]
    
    repli, n, _ = sim_matrices.shape
    
    # Prepare observed data
    pobs = (obs / obs.sum()).flatten()
    nb = np.sum(obs)
    T_range = np.max(obs) - np.min(obs)
    
    # Calculate distance indices for CPCd
    indices = np.floor(distance / 2).astype(int).flatten()
    max_index = indices.max() + 1
    CDD_R = np.bincount(indices, weights=obs.flatten(), minlength=max_index)
    
    results = []
    
    for r in range(repli):
        S = sim_matrices[r]
        result_dict = {"Replication": r}
        
        if "CPC" in measures:
            # CPC - Common Part of Commuters
            mask = (obs != 0) * (S != 0)
            cpc = np.minimum(obs[mask], S[mask]).sum() / nb if nb > 0 else 0
            result_dict["CPC"] = cpc
        
        if "CPL" in measures:
            # CPL - Common Part of Links
            nbNL = ((obs == 0) * (S != 0)).sum()  # Number of new links
            nbML = ((obs != 0) * (S == 0)).sum()  # Number of missing links
            nbCL = ((obs != 0) * (S != 0)).sum()  # Number of common links
            cpl = 2 * nbCL / (nbNL + 2 * nbCL + nbML) if (nbNL + 2 * nbCL + nbML) > 0 else 0
            result_dict["CPL"] = cpl
        
        if "CPCd" in measures:
            # CPCd - Common Part of Commuters by distance
            CDD_S = np.bincount(indices, weights=S.flatten(), minlength=max_index)
            cpcd = (np.abs(CDD_S - CDD_R) / nb).sum() if nb > 0 else 0
            cpcd = 1 - 0.5 * cpcd
            result_dict["CPCd"] = cpcd
        
        if "KS_stat" in measures or "KS_pval" in measures:
            # KS - Kolmogorov-Smirnov test
            ks_statistic, ks_pvalue = _ks_weighted(
                data1=distance.flatten(), 
                wei1=obs.flatten(), 
                wei2=S.flatten()
            )
            if "KS_stat" in measures:
                result_dict["KS_stat"] = ks_statistic
            if "KS_pval" in measures:
                result_dict["KS_pval"] = ks_pvalue
        
        if "KL_div" in measures:
            # KL - Kullback-Leibler divergence
            ppred = (S / S.sum()).flatten() if S.sum() > 0 else S.flatten()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kl_div_array = pobs * np.log(pobs / ppred)
                kl_div = np.nan_to_num(kl_div_array, nan=0., posinf=0., neginf=0.).sum()
            result_dict["KL_div"] = kl_div
        
        if "RMSE" in measures:
            # NRMSE - Normalized Root Mean Square Error
            if T_range > 0 and obs.sum() > 0:
                mse = np.sum((obs - S) ** 2) / obs.sum()
                nrmse = np.sqrt(mse)
            else:
                nrmse = 0
            result_dict["RMSE"] = nrmse
        
        results.append(result_dict)
    
    return pd.DataFrame(results)

def _calculate_gof_shared(params):
    """Calculate goodness-of-fit measures"""
    shm_info, sim_matrices, measures = params
    attached_shms = []
    try:
        # Reconstruct all necessary arrays from shared memory metadata
        obs, shm_obs = _shm_to_numpy(shm_info['obs'])
        distance, shm_d = _shm_to_numpy(shm_info['distance'])
        # Keep track of all attached blocks to ensure they are closed
        attached_shms = [s for s in [shm_obs, shm_d] if s is not None]

        # Ensure sim_matrices is 3D (replications, n, n)
        if sim_matrices.ndim == 2:
            sim_matrices = sim_matrices[np.newaxis, ...]
        
        repli, n, _ = sim_matrices.shape
        
        # Prepare observed data
        pobs = (obs / obs.sum()).flatten()
        nb = np.sum(obs)
        T_range = np.max(obs) - np.min(obs)
        
        # Calculate distance indices for CPCd
        indices = np.floor(distance / 2).astype(int).flatten()
        max_index = indices.max() + 1
        CDD_R = np.bincount(indices, weights=obs.flatten(), minlength=max_index)
        
        results = []
        
        for r in range(repli):
            S = sim_matrices[r]
            result_dict = {"Replication": r}
            
            if "CPC" in measures:
                # CPC - Common Part of Commuters
                mask = (obs != 0) * (S != 0)
                cpc = np.minimum(obs[mask], S[mask]).sum() / nb if nb > 0 else 0
                result_dict["CPC"] = cpc
            
            if "CPL" in measures:
                # CPL - Common Part of Links
                nbNL = ((obs == 0) * (S != 0)).sum()  # Number of new links
                nbML = ((obs != 0) * (S == 0)).sum()  # Number of missing links
                nbCL = ((obs != 0) * (S != 0)).sum()  # Number of common links
                cpl = 2 * nbCL / (nbNL + 2 * nbCL + nbML) if (nbNL + 2 * nbCL + nbML) > 0 else 0
                result_dict["CPL"] = cpl
            
            if "CPCd" in measures:
                # CPCd - Common Part of Commuters by distance
                CDD_S = np.bincount(indices, weights=S.flatten(), minlength=max_index)
                cpcd = (np.abs(CDD_S - CDD_R) / nb).sum() if nb > 0 else 0
                cpcd = 1 - 0.5 * cpcd
                result_dict["CPCd"] = cpcd
            
            if "KS_stat" in measures or "KS_pval" in measures:
                # KS - Kolmogorov-Smirnov test
                ks_statistic, ks_pvalue = _ks_weighted(
                    data1=distance.flatten(), 
                    wei1=obs.flatten(), 
                    wei2=S.flatten()
                )
                if "KS_stat" in measures:
                    result_dict["KS_stat"] = ks_statistic
                if "KS_pval" in measures:
                    result_dict["KS_pval"] = ks_pvalue
            
            if "KL_div" in measures:
                # KL - Kullback-Leibler divergence
                ppred = (S / S.sum()).flatten() if S.sum() > 0 else S.flatten()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kl_div_array = pobs * np.log(pobs / ppred)
                    kl_div = np.nan_to_num(kl_div_array, nan=0., posinf=0., neginf=0.).sum()
                result_dict["KL_div"] = kl_div
            
            if "RMSE" in measures:
                # NRMSE - Normalized Root Mean Square Error
                if T_range > 0 and obs.sum() > 0:
                    mse = np.sum((obs - S) ** 2) / obs.sum()
                    nrmse = np.sqrt(mse)
                else:
                    nrmse = 0
                result_dict["RMSE"] = nrmse
            
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    finally:
        # CRITICAL: Each worker must close its connection to the shared memory blocks
        for shm in attached_shms:
            shm.close()

def _single_metric_average(exponent, params):
    # Specifying explicit boundaries (0, x) results in minimize_scalar getting stuck on the upper bound x
    law, model, measure, mi, mj, distance, opportunity, Oi, Dj, obs, average, repli, processes, verbose = params
    _vprint(f'Trying β = {exponent}', verbose)
    if exponent <=0:
        return np.inf
        
    pij = _proba(law, distance, opportunity, mi, mj, exponent)

    if measure in ['CPC', 'CPL', 'CPCd']:
        sign = -1
    else:
        sign = 1
        	
	# Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    if repli > 1 and num_processes > 1:
        # Parallel processing for multiple realizations
        with mp.Pool(processes=num_processes) as pool:
            _vprint(f'Calculating average {measure} over {repli} realizations', verbose)
            _vprint(f'Using {num_processes} parallel processes', verbose)
            params = [(model, measure, obs, pij, distance, Oi, Dj, average) for r in range(repli)]
            results = list(tqdm(pool.imap(_single_metric, params), total=repli, desc='Computing GOF measure', disable=not verbose))		
            	

        return sign * np.mean(results)
    else:
		# Sequential processing
        if repli > 1:
            result = _single_metric(params) 
            return sign * result
        else:
            results = [_single_metric(params) for r in repli]
            return sign * np.mean(results)

def _single_metric(params):
    model, measure, obs, pij, distance, out_trips, in_trips, average = params
    n = obs.shape[0]
    
    #Simulated T_ij
    S = np.zeros((n, n))
    if model == "UM":
        S = _UM(pij, out_trips, average)
    elif model == "PCM":
        S = _PCM(pij, out_trips, average)
    elif model == "ACM":
        S = _ACM(pij, in_trips, average)
    elif model == "DCM":
        S = _DCM(pij, out_trips, in_trips, 50, 0.01, average)
		
    #Calculate goodness-of-fit measure
    nb = np.sum(obs)
    if measure == "CPC":
        # CPC - Common Part of Commuters
        mask = (obs != 0) * (S != 0)
        cpc = np.minimum(obs[mask], S[mask]).sum() / nb if nb > 0 else 0
        return cpc

    if measure == "CPL":
        # CPL - Common Part of Links
        nbNL = ((obs == 0) * (S != 0)).sum()  # Number of new links
        nbML = ((obs != 0) * (S == 0)).sum()  # Number of missing links
        nbCL = ((obs != 0) * (S != 0)).sum()  # Number of common links
        cpl = 2 * nbCL / (nbNL + 2 * nbCL + nbML) if (nbNL + 2 * nbCL + nbML) > 0 else 0
        return cpl
    if measure == "CPCd":
        # CPCd - Common Part of Commuters by distance
        indices = np.floor(distance / 2).astype(int).flatten()
        max_index = indices.max() + 1
        CDD_R = np.bincount(indices, weights=obs.flatten(), minlength=max_index)
        CDD_S = np.bincount(indices, weights=S.flatten(), minlength=max_index)
        cpcd = (np.abs(CDD_S - CDD_R) / nb).sum() if nb > 0 else 0
        cpcd = 1 - 0.5 * cpcd
        return cpcd
		
    if measure == "KS_stat" or measure == "KS_pval":
        # KS - Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = _ks_weighted(
            data1=distance.flatten(), 
            wei1=obs.flatten(), 
            wei2=S.flatten()
        )
        if measure == "KS_stat":
            return ks_statistic
        else:
            return ks_pvalue

    if measure == "KL_div":
        # KL - Kullback-Leibler divergence
        pobs = (obs / obs.sum()).flatten()
        ppred = (S / S.sum()).flatten() if S.sum() > 0 else S.flatten()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kl_div_array = pobs * np.log(pobs / ppred)
            kl_div = np.nan_to_num(kl_div_array, nan=0., posinf=0., neginf=0.).sum()
            return kl_div

    if measure == "RMSE":
        # NRMSE - Normalized Root Mean Square Error
        T_range = np.max(obs) - np.min(obs)
        if T_range > 0 and obs.sum() > 0:
            mse = np.sum((obs - S) ** 2) / obs.sum()
            nrmse = np.sqrt(mse)
        else:
            nrmse = 0
        return nrmse
            
def _proba(law, dij, sij, mi, mj, beta):
    """Generate the matrix pij according to the law"""
    n = len(mi)
    W = np.zeros((n, n))

    if law == "GravExp":
        W = np.outer(mi, mj) * np.exp(-dij * beta)
        np.fill_diagonal(W, 0)

    elif law == "NGravExp":
        W = mj * np.exp(-dij * beta)
        np.fill_diagonal(W, 0)

    elif law == "GravPow":
        W = np.outer(mi, mj) * dij**(-beta)
        np.fill_diagonal(W, 0)

    elif law == "NGravPow":
        W = mj * dij**(-beta)
        np.fill_diagonal(W, 0)

    elif law == "Schneider":
        W = np.exp(-beta * sij) - np.exp(-beta * (sij + mj))
        np.fill_diagonal(W, 0)
        W[np.isnan(W)] = 0

    elif law == "Rad":
        W = np.outer(mi, mj) / ((mi[:, np.newaxis] + sij) * (mi[:, np.newaxis] + mj + sij))
        np.fill_diagonal(W, 0)
        W[np.isnan(W)] = 0

    elif law == "RadExt":
        numerator = ((mi[:, np.newaxis] + mj + sij)**beta - (mi[:, np.newaxis] + sij)**beta) * (mi**beta + 1)[:, np.newaxis]
        denominator = ((mi[:, np.newaxis] + mj + sij)**beta + 1) * ((mi[:, np.newaxis] + sij)**beta + 1)
        W = numerator / denominator
        np.fill_diagonal(W, 0)
        W[np.isnan(W)] = 0

    elif law == "Rand":
        W = np.ones((n, n)) / (n**2 - n)
        np.fill_diagonal(W, 0)

    # Row normalization if needed
    if law not in ["GravExp", "GravPow", "Rand"]:
        Wi = np.sum(W, axis=1)
        mask = Wi != 0
        W[mask] = mi[mask, np.newaxis] * W[mask] / Wi[mask, np.newaxis]

    return W


def _UM(pij, Oi, average):
    """Generate the network using the Unconstrained Model"""
    n = pij.shape[0]
    nb_commuters = np.sum(Oi)
    sumt = np.sum(pij)
    sum_rows = np.sum(pij, axis=1)
    
    if not average:
        S = np.floor(nb_commuters * pij / sumt) if sumt > 0 else np.zeros_like(pij)
        nb = np.sum(S)
        
        remaining = int(nb_commuters - nb)
        if remaining > 0:
            index = _Multinomial_ij(remaining, pij, sum_rows)
            flat_indices = index[:, 0] * n + index[:, 1]
            increments = np.bincount(flat_indices, minlength=n*n).reshape(n, n)
            S += increments
    else:
        S = nb_commuters * pij / sumt
        
    return S


def _PCM(pij, Oi, average):
    """Generate the network using the Production Constrained Model"""
    n = len(Oi)
    S = np.zeros((n, n))
    sum_rows = np.sum(pij, axis=1)

    # Initial allocation
    valid_rows = sum_rows > 0
    division_factors = np.zeros(n)
    division_factors[valid_rows] = 1.0 / sum_rows[valid_rows]
    
    allocation_ratios = pij * division_factors[:, np.newaxis]
    
    if not average:
        S = np.floor(Oi[:, np.newaxis] * allocation_ratios)
        
        # Allocate remaining commuters
        nb = np.sum(S, axis=1).astype(int)
        
        for i in range(n):
            remaining = Oi[i] - nb[i]
            if remaining > 0 and sum_rows[i] > 0:
                index = _Multinomial_i(remaining, pij[i], sum_rows[i])
                increments = np.bincount(index, minlength=n)
                S[i] += increments
    else:
        S = Oi[:, np.newaxis] * allocation_ratios
        
    return S


def _ACM(pij, Dj, average):
    """Generate the network using the Attraction Constrained Model"""
    n = len(Dj)
    S = np.zeros((n, n))
    tweights = pij.T
    sum_rows = np.sum(tweights, axis=1)

    # Initial allocation
    valid_rows = sum_rows > 0
    division_factors = np.zeros(n)
    division_factors[valid_rows] = 1.0 / sum_rows[valid_rows]
    
    allocation_ratios = tweights * division_factors[:, np.newaxis]
    
    if not average:
        initial_allocation = np.floor(Dj[:, np.newaxis] * allocation_ratios)
        S = initial_allocation.T
        
        # Allocate remaining commuters
        nb = np.sum(S, axis=0).astype(int)
        
        for i in range(n):
            remaining = Dj[i] - nb[i]
            if remaining > 0 and sum_rows[i] > 0:
                index = _Multinomial_i(remaining, tweights[i], sum_rows[i])
                increments = np.bincount(index, minlength=n)
                S[:, i] += increments
    else:
        initial_allocation = Dj[:, np.newaxis] * allocation_ratios
        S = initial_allocation.T
    
    return S


def _DCM(pij, Oi, Dj, max_iter, closure, average):
    """Generate the network using the Doubly Constrained Model"""
    n = len(Oi)
    
    # Initialize marginals
    marg = np.zeros((n, 2))
    marg[:, 0] = np.maximum(Oi, 0.01)
    marg[:, 1] = np.maximum(Dj, 0.01)
    
    weights = np.maximum(pij, 0.01)
    
    iter_count = 0
    crit_out = 1.0
    crit_in = 1.0
    
    # IPF procedure
    while (crit_out > closure or crit_in > closure) and (iter_count <= max_iter):
        # Row adjustment
        sout = np.sum(weights, axis=1)
        adjustment_factors = marg[:, 0] / sout
        weights = weights * adjustment_factors[:, np.newaxis]
        
        # Column adjustment
        sin = np.sum(weights, axis=0)
        adjustment_factors = marg[:, 1] / sin
        weights = weights * adjustment_factors[np.newaxis, :]
        
        # Check convergence
        sout = np.sum(weights, axis=1)
        sin = np.sum(weights, axis=0)
        
        rel_error_out = np.abs(1 - (sout / marg[:, 0]))
        rel_error_in = np.abs(1 - (sin / marg[:, 1]))
        
        crit_out = np.max(rel_error_out)
        crit_in = np.max(rel_error_in)
        
        iter_count += 1

    # Generate final matrix using UM
    if not average:
        S = _UM(weights, Oi, average)
    else:
        S = weights
        
    return S


def _Multinomial_i(n, weights, sum_val):
    """Sample indices according to weights"""
    n = int(n)
    if n <= 0 or sum_val <= 0:
        return np.array([], dtype=int)
        
    random_vals = np.random.random(n) * sum_val
    cumulative_weights = np.cumsum(weights)
    random_index = np.searchsorted(cumulative_weights, random_vals)
    
    return random_index


def _Multinomial_ij(n, weights, sum_rows):
    """Sample 2D indices according to matrix weights"""
    n = int(n)
    if n <= 0:
        return np.array([]).reshape(0, 2)
        
    sumt = np.sum(sum_rows)
    if sumt <= 0:
        return np.array([]).reshape(0, 2)
        
    random_vals = np.random.random(n) * sumt
    cumsum_rows = np.cumsum(sum_rows)
    row_indices = np.searchsorted(cumsum_rows, random_vals)
    
    # Calculate remaining values for column selection
    prev_cumsum = np.zeros(n)
    mask = row_indices > 0
    prev_cumsum[mask] = cumsum_rows[row_indices[mask] - 1]
    remaining_vals = random_vals - prev_cumsum
    
    # Select columns
    col_indices = np.zeros(n, dtype=int)
    for i in range(n):
        row = row_indices[i]
        if row < weights.shape[0]:
            row_weights = weights[row]
            cumsum_cols = np.cumsum(row_weights)
            col_indices[i] = np.searchsorted(cumsum_cols, remaining_vals[i])
    
    return np.column_stack((row_indices, col_indices))


def _pkstwo(x, tol=0):
    """Calculate CDF of Kolmogorov-Smirnov two-sample test statistic"""
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.asarray(x)
    
    p = np.zeros_like(x, dtype=float)
    p[np.isnan(x)] = np.nan
    
    idx = np.where(~np.isnan(x) & (x > 0))[0]
    
    for i in idx:
        if x[i] < 1e-10:
            p[i] = 0.0
        else:
            k_max = int(np.ceil(45 / x[i]**2))
            sum_term = 0.0
            for k in range(1, k_max + 1):
                t1 = np.exp(-2 * k**2 * x[i]**2)
                t2 = 2 * k**2 * x[i]**2 - 1
                sum_term += t1 * t2
                if t1 * t2 < tol:
                    break
            p[i] = 1 - 2 * sum_term
    
    if len(p) == 1:
        return p[0]
    return p


def _ks_weighted(data1, data2=None, wei1=None, wei2=None, alternative='two-sided'):
    """Compute Kolmogorov-Smirnov statistic for weighted data"""
    if data2 is None:
        data2 = data1
        
    ix1 = np.argsort(data1)
    data1_sorted = data1[ix1]
    wei1_sorted = wei1[ix1]
    
    if data1 is data2:
        data2_sorted = data1_sorted
        ix2 = ix1
    else:
        ix2 = np.argsort(data2)
        data2_sorted = data2[ix2]
    
    wei2_sorted = wei2[ix2]
    
    # Calculate CDFs
    if np.array_equal(data1_sorted, data2_sorted):
        cwei1 = np.hstack([0, np.cumsum(wei1_sorted) / np.sum(wei1_sorted)])
        cwei2 = np.hstack([0, np.cumsum(wei2_sorted) / np.sum(wei2_sorted)])
        cdf1we = cwei1[1:]
        cdf2we = cwei2[1:]
    else:
        data = np.concatenate([data1_sorted, data2_sorted])
        cwei1 = np.hstack([0, np.cumsum(wei1_sorted) / np.sum(wei1_sorted)])
        cwei2 = np.hstack([0, np.cumsum(wei2_sorted) / np.sum(wei2_sorted)])
        cdf1we = cwei1[np.searchsorted(data1_sorted, data, side='right')]
        cdf2we = cwei2[np.searchsorted(data2_sorted, data, side='right')]
    
    # Calculate KS statistic
    if alternative == 'two-sided':
        d = np.max(np.abs(cdf1we - cdf2we))
    elif alternative == 'less':
        d = np.max(cdf2we - cdf1we)
    elif alternative == 'greater':
        d = np.max(cdf1we - cdf2we)
    else:
        raise ValueError("alternative must be one of 'two-sided', 'less', or 'greater'")
    
    # Calculate effective sample sizes
    n1_effective = np.sum(wei1)**2 / np.sum(wei1**2) if np.sum(wei1**2) > 0 else 0
    n2_effective = np.sum(wei2)**2 / np.sum(wei2**2) if np.sum(wei2**2) > 0 else 0
    
    if n1_effective > 0 and n2_effective > 0:
        n_effective = (n1_effective * n2_effective) / (n1_effective + n2_effective)
    else:
        n_effective = 0
    
    # Calculate p-value
    if alternative == 'two-sided' and n_effective > 0:
        prob = 1 - _pkstwo(np.sqrt(n_effective) * d)
    else:
        prob = np.nan
    
    return d, prob

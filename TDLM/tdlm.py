"""
TDLM: Trip Distribution Law Models Library

Author: Maxime Lenormand (2015)
Converted to library format with enhanced parallel processing support

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
from typing import Union, Optional, List, Dict
import warnings


class TDLMError(Exception):
    """Custom exception for TDLM errors"""
    pass


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
    repli: int = 1,
    processes: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run trip distribution law model simulations.
    
    Parameters
    ----------
    law : str
        Trip distribution law. One of: "GravExp", "NGravExp", "GravPow", 
        "NGravPow", "Schneider", "Rad", "RadExt", "Rand"
    mass_origin : np.ndarray
        Number of inhabitants at origin (mi)
    mass_destination : np.ndarray  
        Number of inhabitants at destination (mj)
    distance : np.ndarray
        Distance matrix (n x n)
    opportunity : np.ndarray, optional
        Matrix of opportunities (n x n). Required for "Rad", "RadExt", "Schneider"
    exponent : float or np.ndarray
        Exponent parameter(s) for the distribution law
    return_proba : bool, default False
        Whether to return probability matrices
    model : str, default "UM"
        Distribution model. One of: "UM", "PCM", "ACM", "DCM"
    out_trips : np.ndarray, optional
        Number of out-commuters (Oi). Required for constrained models
    in_trips : np.ndarray, optional
        Number of in-commuters (Dj). Required for ACM and DCM models
    repli : int, default 1
        Number of replications
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Union[np.ndarray, Dict[str, np.ndarray]]
        If single exponent: np.ndarray of shape (repli, n, n)
        If multiple exponents: Dict with exponents as keys, arrays as values
    """
    
    # Input validation
    _validate_inputs(law, model, mass_origin, mass_destination, distance, 
                    opportunity, out_trips, in_trips)
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Handle single vs multiple exponents
    exponents = np.atleast_1d(exponent)
    single_exponent = len(exponents) == 1
    
    # Setup data tuple
    n = len(mass_origin)
    data = (n, mass_origin, mass_destination, out_trips, in_trips, distance, opportunity)
    
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    if len(exponents) > 1 and num_processes > 1:
        # Parallel processing for multiple exponents
        print(f'Running simulations for {law} with {model} model ({repli} replications)')
        print(f'Using {num_processes} parallel processes')
        
        with mp.Pool(processes=num_processes) as pool:
            params = [(data, law, model, beta, repli, return_proba) for beta in exponents]
            results = list(tqdm(pool.imap(_process_exponent, params), 
                              total=len(exponents), desc='Computing exponents'))
        
        # Organize results
        output = {}
        for i, beta in enumerate(exponents):
            if return_proba:
                output[beta] = results[i]
            else:
                output[beta] = results[i]['simulations']
                
    else:
        output = {}
        # Sequential processing
        if single_exponent:
            beta = exponents[0]
            print(f'Simulating matrix for {law} β = {beta:.2g} with {model}')
            params = (data, law, model, beta, repli, return_proba)
            result = _process_exponent(params)
            if return_proba:
                output[beta] = result
            else:
                output[beta] = result['simulations']
        else:
            print(f'Running simulations for {law} with {model} model ({repli} replications)')
            
            
            for i, beta in enumerate(tqdm(exponents, desc='Computing exponents')):
                params = (data, law, model, beta, repli, return_proba)
                result = _process_exponent(params)
                if return_proba:
                    output[beta] = result
                else:
                    output[beta] = result['simulations']
    print('Done\n')
    
    # Return format based on input
    if single_exponent:
        return list(output.values())[0]
    else:
        return output


def gof(
    sim: Union[np.ndarray, Dict[str, np.ndarray]], 
    obs: np.ndarray,
    distance: np.ndarray,
    measures: Union[str, List[str]] = "all",
    processes: Optional[int] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Calculate goodness-of-fit measures for simulated vs observed trip matrices.
    
    Parameters
    ----------
    sim : np.ndarray or Dict[str, np.ndarray]
        Simulated trip matrices. If Dict, keys should be exponent values
    obs : np.ndarray
        Observed trip matrix (n x n)
    distance : np.ndarray
        Distance matrix (n x n)
    measures : str or List[str], default "all"
        Measures to calculate. "all" or subset of:
        ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    processes : int, optional
        Number of processes for parallel computation. Default: CPU count - 2
        
    Returns
    -------
    Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        If single simulation: DataFrame with measures
        If multiple simulations: Dict with exponents as keys, DataFrames as values
    """
    
    # Available measures
    all_measures = ["CPC", "CPL", "CPCd", "KS_stat", "KS_pval", "KL_div", "RMSE"]
    
    if measures == "all":
        selected_measures = all_measures
    else:
        selected_measures = measures if isinstance(measures, list) else [measures]
        invalid = set(selected_measures) - set(all_measures)
        if invalid:
            raise TDLMError(f"Invalid measures: {invalid}. Available: {['all']+all_measures}")
    
    # Setup multiprocessing
    num_processes = processes if processes is not None else max(1, mp.cpu_count() - 2)
    
    # Handle single vs multiple simulations
    if isinstance(sim, dict):
        exponents = list(sim.keys())
        single_simulation = len(exponents) == 1
        
        if len(exponents) > 1 and num_processes > 1:
            # Parallel processing for multiple exponents
            print(f'Calculating GOF measures for {len(exponents)} exponents')
            print(f'Using {num_processes} parallel processes')
            
            with mp.Pool(processes=num_processes) as pool:
                params = [(exponent, sim_matrices, obs, distance, selected_measures) 
                         for exponent, sim_matrices in sim.items()]
                results = list(tqdm(pool.imap(_process_gof_exponent, params), 
                                  total=len(exponents), desc='Computing GOF measures'))
            
            # Organize results
            output = {}
            for i, exponent in enumerate(exponents):
                output[exponent] = results[i]
            
            print('Done\n')
            return output
        else:
            # Sequential processing
            results = {}
            if single_simulation:
                exponent = exponents[0]
                print(f'Calculating GOF measures for exponent {exponent}')
                results[exponent] = _calculate_gof(sim[exponent], obs, distance, selected_measures)
            else:
                print(f'Calculating GOF measures for {len(exponents)} exponents')
                for exponent in tqdm(exponents, desc='Computing GOF measures'):
                    results[exponent] = _calculate_gof(sim[exponent], obs, distance, selected_measures)
            
            print('Done\n')
            return results
    else:
        # Single simulation matrix
        print('Calculating GOF measures')
        result = _calculate_gof(sim, obs, distance, selected_measures)
        print('Done\n')
        return result


def _process_gof_exponent(params):
    """Process GOF calculation for a single exponent"""
    exponent, sim_matrices, obs, distance, selected_measures = params
    return _calculate_gof(sim_matrices, obs, distance, selected_measures)



def _validate_inputs(law, model, mass_origin, mass_destination, distance, 
                    opportunity, out_trips, in_trips):
    """Validate input parameters"""
    
    valid_laws = ["GravExp", "NGravExp", "GravPow", "NGravPow", "Schneider", "Rad", "RadExt", "Rand"]
    valid_models = ["UM", "PCM", "ACM", "DCM"]
    
    if law not in valid_laws:
        raise TDLMError(f"Invalid law '{law}'. Must be one of: {valid_laws}")
    
    if model not in valid_models:
        raise TDLMError(f"Invalid model '{model}'. Must be one of: {valid_models}")
    
    # Check array dimensions
    n = len(mass_origin)
    if len(mass_destination) != n:
        raise TDLMError("mass_origin and mass_destination must have same length")
    
    if distance.shape != (n, n):
        raise TDLMError(f"distance matrix must be {n}x{n}")
    
    # Check opportunity matrix for relevant laws
    if law in ["Rad", "RadExt", "Schneider"]:
        if opportunity is None:
            raise TDLMError(f"opportunity matrix required for law '{law}'")
        if opportunity.shape != (n, n):
            raise TDLMError(f"opportunity matrix must be {n}x{n}")
    
    # Check trip constraints for models
    if model in ["PCM", "DCM"] and out_trips is None:
        raise TDLMError(f"out_trips required for model '{model}'")
    
    if model in ["ACM", "DCM"] and in_trips is None:
        raise TDLMError(f"in_trips required for model '{model}'")
    
    if out_trips is not None and len(out_trips) != n:
        raise TDLMError("out_trips must have same length as mass arrays")
        
    if in_trips is not None and len(in_trips) != n:
        raise TDLMError("in_trips must have same length as mass arrays")


def _process_exponent(params):
    """Process a single exponent value"""
    data, law, model, beta, repli, return_proba = params
    n, mi, mj, Oi, Dj, dij, sij = data
    
    # Build the matrix pij according to the law
    pij = _proba(law, dij, sij, mi, mj, beta)
    
    # Store results
    simulations = []
    
    # Loop replications
    for r in range(repli):
        # Simulated OD
        S = np.zeros((n, n))

        # Network generation according to the constrained model
        if model == "UM":  # Unconstrained model
            S = _UM(pij, Oi)
        elif model == "PCM":  # Production constrained model
            S = _PCM(pij, Oi)
        elif model == "ACM":  # Attraction constrained model
            S = _ACM(pij, Dj)
        elif model == "DCM":  # Doubly constrained model
            S = _DCM(pij, Oi, Dj, 50, 0.01)

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


def _calculate_gof(sim_matrices, obs, distance, measures):
    """Calculate goodness-of-fit measures"""
    
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


# Import the utility functions from the original scripts
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


def _UM(pij, Oi):
    """Generate the network using the Unconstrained Model"""
    n = pij.shape[0]
    nb_commuters = np.sum(Oi)
    sumt = np.sum(pij)
    sum_rows = np.sum(pij, axis=1)

    S = np.floor(nb_commuters * pij / sumt) if sumt > 0 else np.zeros_like(pij)
    nb = np.sum(S)
    
    remaining = int(nb_commuters - nb)
    if remaining > 0:
        index = _Multinomial_ij(remaining, pij, sum_rows)
        flat_indices = index[:, 0] * n + index[:, 1]
        increments = np.bincount(flat_indices, minlength=n*n).reshape(n, n)
        S += increments

    return S


def _PCM(pij, Oi):
    """Generate the network using the Production Constrained Model"""
    n = len(Oi)
    S = np.zeros((n, n))
    sum_rows = np.sum(pij, axis=1)

    # Initial allocation
    valid_rows = sum_rows > 0
    division_factors = np.zeros(n)
    division_factors[valid_rows] = 1.0 / sum_rows[valid_rows]
    
    allocation_ratios = pij * division_factors[:, np.newaxis]
    S = np.floor(Oi[:, np.newaxis] * allocation_ratios)
    
    # Allocate remaining commuters
    nb = np.sum(S, axis=1).astype(int)
    
    for i in range(n):
        remaining = Oi[i] - nb[i]
        if remaining > 0 and sum_rows[i] > 0:
            index = _Multinomial_i(remaining, pij[i], sum_rows[i])
            increments = np.bincount(index, minlength=n)
            S[i] += increments
    
    return S


def _ACM(pij, Dj):
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
    
    return S


def _DCM(pij, Oi, Dj, max_iter, closure):
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
    S = _UM(weights, Oi)
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
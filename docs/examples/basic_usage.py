"""
Basic Usage Example for TDLM Library
====================================

This example demonstrates how to use the TDLM library for trip distribution modeling.
"""
import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TDLM import tdlm

data_path = os.getcwd() + os.sep + "docs" + os.sep + "examples" + os.sep + "data_US" + os.sep

def main():
    """Main example function"""
    
    # Sample data - replace with your actual data
    print("Setting up example data...")
    
    #Inputs (mi, mj, Oi and Dj)
    inputs = pd.read_csv(data_path + "Inputs.csv", sep=";")

    # Origin and destination masses (population, employment, etc.)
    mi = inputs.iloc[:, 0].values  # Number of inhabitants at origin (mi)
    mj = inputs.iloc[:, 1].values  # Number of inhabitants at destination (mj)
    
    # Trip constraints (for constrained models)
    Oi = inputs.iloc[:, 2].values.astype(int)  # Number of out-commuters (Oi)
    Dj = inputs.iloc[:, 3].values.astype(int)  # Number of in-commuters (Dj)

    # Distance matrix dij (size n x n)
    # Locale detection (whether decimal place is marked by "." or ",")
    decimal = '.'
    with gzip.open(data_path + "Distance.csv.gz", 'rt') as f:
        _ = f.readline() #discard header
        for line in f:
            if ',' in line:
                decimal = ','
                break
            
    dij = pd.read_csv(data_path + "Distance.csv.gz", compression='gzip', sep=";", decimal=decimal, header=0)
    dij = dij.values.astype(float)
    
    # Observed OD matrix Tij (size n x n)
    Tij_df = pd.read_csv(data_path + "OD.csv", sep=";", header=0)
    Tij_observed = Tij_df.values.astype(float)
    
    print("Running TDLM simulation...")
    
    # Define law and model
    law = 'NGravExp'  # Normalized Gravity with Exponential decay
    model = 'DCM'     # Doubly Constrained Model
    
    # Test range of exponent values
    exponent = np.arange(0.01, 0.15, 0.005).round(3)
    
    # Run the simulation
    results = tdlm.run_law_model(
        law=law,
        mass_origin=mi,
        mass_destination=mj,
        distance=dij,
        opportunity=None,        # Not needed for this law
        exponent=exponent,
        return_proba=False,
        model=model,
        out_trips=Oi,
        in_trips=Dj,
        repli=5,               # Number of replications
        random_seed=42          # For reproducibility
    )
    
    print("Calculating goodness-of-fit measures...")
    
    # Calculate goodness-of-fit
    gof_results = tdlm.gof(
        sim=results,
        obs=Tij_observed,
        distance=dij,
        measures="all"
    )
    
    # Process results for plotting
    print("Processing results...")
    
    # Calculate mean and std for each exponent
    metrics_summary = {}
    for exp in exponent:
        df = gof_results[exp]
        metrics_summary[exp] = {
            'CPC_mean': df['CPC'].mean(),
            'CPC_std': df['CPC'].std(),
            'CPL_mean': df['CPL'].mean(),
            'CPL_std': df['CPL'].std(),
            'CPCd_mean': df['CPCd'].mean(),
            'CPCd_std': df['CPCd'].std(),
            'KS_stat_mean': df['KS_stat'].mean(),
            'KS_stat_std': df['KS_stat'].std(),
            'KL_div_mean': df['KL_div'].mean(),
            'KL_div_std': df['KL_div'].std(),
            'RMSE_mean': df['RMSE'].mean(),
            'RMSE_std': df['RMSE'].std(),
        }
    
    # Plot results
    print("Creating plots...")
    
    list_metrics = ['CPC', 'CPL', 'CPCd', 'KS_stat', 'KL_div', 'RMSE']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(list_metrics):
        means = [metrics_summary[exp][f'{metric}_mean'] for exp in exponent]
        stds = [metrics_summary[exp][f'{metric}_std'] for exp in exponent]
        
        axes[i].errorbar(exponent, means, yerr=stds, fmt='-o', 
                        linewidth=0.75, elinewidth=1, capsize=2, 
                        capthick=1, markerfacecolor='none')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('Exponent')
        axes[i].grid(True, alpha=0.3)
        
        # Find optimal exponent (highest value for CPC, CPL, CPCd; lowest for others)
        if metric in ['CPC', 'CPL', 'CPCd']:
            optimal_idx = np.argmax(means)
        else:
            optimal_idx = np.argmin(means)
        
        axes[i].axvline(exponent[optimal_idx], color='red', linestyle='--', 
                       alpha=0.7, label=f'Optimal: {exponent[optimal_idx]:.3g}')
        axes[i].legend()
    
    fig.suptitle(f'{law} Model with {model} Constraints', fontsize=16)
    fig.tight_layout()
    fig.show()
    
    # Print best exponent based on CPC
    cpc_means = [metrics_summary[exp]['CPC_mean'] for exp in exponent]
    best_exp_idx = np.argmax(cpc_means)
    best_exponent = exponent[best_exp_idx]
    
    print("\nResults Summary:")
    print(f"Law: {law}")
    print(f"Model: {model}")
    print(f"Best exponent (highest CPC): {best_exponent:.3g}")
    print(f"Best CPC value: {cpc_means[best_exp_idx]:.3f}")
    
    # Show sample simulation with best exponent
    print(f"\nSample simulation with optimal exponent ({best_exponent}):")
    best_sim = tdlm.run_law_model(
        law=law,
        mass_origin=mi,
        mass_destination=mj,
        distance=dij,
        exponent=best_exponent,
        model=model,
        out_trips=Oi,
        in_trips=Dj,
        repli=1,
        return_proba=True,
        random_seed=42
    )
    best_sim_gof = tdlm.gof(
        sim=best_sim['simulations'],
        obs=Tij_observed,
        distance=dij,
        measures="all"
    )
    print("Observed matrix:")
    print(Tij_observed)
    print("Simulated matrix:")
    print(best_sim['simulations'].squeeze().astype(int))
    print('Metrics:')
    print(best_sim_gof.to_markdown(index=False))
    


if __name__ == "__main__":
    main()

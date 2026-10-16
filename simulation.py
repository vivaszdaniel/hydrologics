"""
=============================================================================
HYDROLOGICS PROTOCOL: Sensorless pH Monitoring via Voltammetric Onset
=============================================================================
Implementation of the HydroLogics protocol for indirect pH estimation
via Hydrogen Evolution Reaction (HER) with zero-mean ohmic fluctuations.

Reference:
  Vivas, D., Sanchez, J., Brito, Y., Duque, J., & Armando, A. (2026).
  "Theoretical Framework for Sensorless pH Monitoring in Hydrometallurgical
  Systems: A Voltammetric Overpotential Approach for Edge Computing."

Authors: D.V. & J.S.
Date: 2026-10-12
License: Open Source / GNU GPL v3.0
=============================================================================
"""

import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import json

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
np.random.seed(42)

# Physical constants
R_GAS = 8.314462618      # J/(mol·K)
F_CONST = 96485.33212    # C/mol
T_KELVIN = 298.15        # K (25°C)

# HydroLogics protocol parameters (Section 6 of paper)
PARAMS = {
    'Ru': 15.0,                       # Ω (uncompensated resistance)
    'sigma_noise': 0.1e-3,            # A (measurement noise)
    'delta_E_ref_std': 15e-3,         # V (pseudo-reference drift standard deviation)
    'K_cal_true': -0.3200,            # V (true physical reference value)
    'V_max': -1.5,                    # V (sweep limit, hydrometallurgy mode)
    'interference_prob': 0.05,        # Redox interference probability (5%)
    'interference_magnitude': 0.030,  # V (interference magnitude)
}

# pH levels and simulation settings
pH_levels = np.linspace(2.0, 12.0, 21)
N_ITERATIONS = 50
S_NERNST = (2.303 * R_GAS * T_KELVIN) / F_CONST  # ~0.05916 V/pH


# ============================================================================
# PHYSICAL MODEL: Vonset Generator (Zero-Mean Ohmic Fluctuation)
# ============================================================================
def generate_Vonset(pH_true, K_cal, noise=True, include_interference=True):
    """
    Generates the onset potential (Vonset) for a given pH, incorporating 
    reference drift, zero-mean ohmic fluctuations, measurement noise, and interferences.
    """
    # Deterministic term (calibrated empirical model)
    Vonset_theoretical = K_cal + (-S_NERNST * pH_true)
    
    # Source 1: Pseudo-reference drift
    delta_E_ref = np.random.normal(0, PARAMS['delta_E_ref_std'])
    
    # Source 2: Ohmic drop fluctuation model (Strictly zero-mean to avoid calibration shift)
    iRu_drop = np.random.normal(0, 0.015)  # V (fluctuación simétrica de media cero)
    
    # Source 3: Gaussian measurement noise
    measurement_noise = np.random.normal(0, 2e-3) if noise else 0.0
    
    # Source 4: Redox interferences (heavy tails)
    interference = 0.0
    if include_interference and np.random.rand() < PARAMS['interference_prob']:
        sign = np.random.choice([-1, 1])
        magnitude = np.random.uniform(0.020, PARAMS['interference_magnitude'])
        interference = sign * magnitude
    
    return Vonset_theoretical + delta_E_ref - iRu_drop + measurement_noise + interference


# ============================================================================
# K_cal CALIBRATION
# ============================================================================
def calibrate_system(buffer_pH_list, n_replicates=3):
    """
    Calibration of empirical parameter K_cal using 5 standard buffers
    (pH 2.0, 4.0, 7.0, 10.0, 12.0) with 3 replicates each.
    """
    V_onset_means = []
    
    for pH_buffer in buffer_pH_list:
        V_onset_replicates = [
            generate_Vonset(pH_buffer, PARAMS['K_cal_true'], 
                          noise=True, include_interference=False)
            for _ in range(n_replicates)
        ]
        V_onset_means.append(np.mean(V_onset_replicates))
    
    # Linear regression: K_cal = Vonset + S_NERNST * pH
    K_cal_values = [V + S_NERNST * pH for V, pH in zip(V_onset_means, buffer_pH_list)]
    K_cal_estimated = np.mean(K_cal_values)
    u_K = np.std(K_cal_values, ddof=1) / np.sqrt(len(K_cal_values))
    
    return K_cal_estimated, u_K


# ============================================================================
# 1-STATE KALMAN FILTER
# ============================================================================
class HydroLogicsFilter:
    """
    1-state Kalman filter for pH estimation.
    """
    
    def __init__(self, K_cal, u_K):
        self.K_cal = K_cal
        
        # Measurement variance in Vonset units (V²)
        self.R_vonset = (u_K)**2 + (0.015)**2 + (0.010)**2
        
        # Initial pH variance
        self.P = 1.0
        
        # Process noise
        self.Q = 1e-4
        
        # Initial estimate
        self.pH_est = 7.0
        
        # Observation model: H = -S_NERNST
        self.H = -S_NERNST
    
    def update(self, Vonset_measured):
        """
        Kalman filter update step.
        """
        # 1. Prediction
        pH_pred = self.pH_est
        P_pred = self.P + self.Q
        
        # 2. Innovation: Vonset_measured - Vonset_expected
        Vonset_expected = self.K_cal + self.H * pH_pred
        innovation = float(Vonset_measured - Vonset_expected)
        
        # 3. Kalman Gain
        S_innov = (self.H**2) * P_pred + self.R_vonset
        K_gain = self.H * P_pred / S_innov
        
        # 4. Update
        self.pH_est = pH_pred + K_gain * innovation
        self.P = (1 - K_gain * self.H) * P_pred
        
        return self.pH_est, self.P


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================
def run_monte_carlo(K_cal_estimated, u_K):
    """
    Runs the complete Monte Carlo simulation.
    """
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION")
    print("="*70)
    print(f"pH levels: {len(pH_levels)}")
    print(f"Iterations per level: {N_ITERATIONS}")
    print(f"Total evaluations: {len(pH_levels) * N_ITERATIONS}")
    print(f"Include redox interferences: {PARAMS['interference_prob']*100:.0f}%")
    
    results = []
    
    for pH_target in pH_levels:
        filter_hl = HydroLogicsFilter(K_cal_estimated, u_K)
        pH_estimates = []
        
        for iteration in range(N_ITERATIONS):
            Vonset_measured = generate_Vonset(
                pH_target, K_cal_estimated,
                noise=True, include_interference=True
            )
            pH_est, _ = filter_hl.update(Vonset_measured)
            pH_estimates.append(pH_est)
        
        results.append({
            'pH_true': pH_target,
            'pH_estimates': np.array(pH_estimates),
            'pH_mean': np.mean(pH_estimates),
            'pH_std': np.std(pH_estimates, ddof=1)
        })
        
        if int(pH_target) == pH_target:
            print(f"  pH {pH_target:.0f}: "
                  f"estimated = {np.mean(pH_estimates):.3f} ± "
                  f"{np.std(pH_estimates, ddof=1):.3f}")
    
    return results


# ============================================================================
# VALIDATION METRICS
# ============================================================================
def calculate_metrics(results):
    """
    Calculates protocol performance metrics.
    """
    pH_true_all = np.array([r['pH_true'] for r in results])
    pH_est_all = np.concatenate([r['pH_estimates'] for r in results])
    pH_true_repeated = np.repeat(pH_true_all, N_ITERATIONS)
    
    residuals = pH_est_all - pH_true_repeated
    
    MSE = np.mean(residuals**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((pH_true_repeated - np.mean(pH_true_repeated))**2)
    R2 = 1 - (SS_res / SS_tot)
    NSE = 1 - (SS_res / SS_tot)
    
    U_pH = 2 * np.std(residuals)
    coverage = np.mean(np.abs(residuals) <= U_pH) * 100
    kurt = kurtosis(residuals, fisher=True)
    
    return {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'bias': bias,
        'R2': R2,
        'NSE': NSE,
        'U_pH': U_pH,
        'coverage': coverage,
        'kurtosis': kurt,
        'residuals': residuals,
        'pH_true': pH_true_all
    }


# ============================================================================
# SEPARATE FIGURES GENERATION
# ============================================================================
def generate_separate_figures(results, metrics, K_cal_estimated, u_K):
    """
    Generates four independent, high-resolution figures in English.
    """
    pH_true = np.array([r['pH_true'] for r in results])
    pH_mean = np.array([r['pH_mean'] for r in results])
    pH_std = np.array([r['pH_std'] for r in results])
    
    # Figure 1: Target pH vs Estimated pH
    plt.figure(figsize=(7, 5), dpi=300)
    plt.fill_between(pH_true, pH_true - metrics['U_pH'], pH_true + metrics['U_pH'],
                     color='#d9e2ec', alpha=0.6, label=f'95% CI ($U_{{pH}} = \\pm{metrics["U_pH"]:.3f}$)')
    plt.plot([2, 12], [2, 12], 'k--', linewidth=1.2, label='Ideal Baseline ($y=x$)')
    plt.errorbar(pH_true, pH_mean, yerr=pH_std, fmt='o', color='#102a43', ecolor='#829ab1',
                 elinewidth=1.2, capsize=3, markersize=5,
                 label=f'HydroLogics ($R^2 = {metrics["R2"]:.4f}$)')
    plt.xlabel('Target pH ($pH_{{true}}$)')
    plt.ylabel('Estimated pH ($\\hat{{pH}}$)')
    plt.title('Monte Carlo Validation: Target vs. Estimated pH')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(1.5, 12.5)
    plt.ylim(1.5, 12.5)
    plt.tight_layout()
    plt.savefig('fig1_ph_target_vs_estimated.png')
    plt.close()

    # Figure 2: Error Distribution
    plt.figure(figsize=(7, 5), dpi=300)
    residuals = metrics['residuals']
    plt.hist(residuals, bins=40, density=True, alpha=0.7, color='#102a43', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(metrics['bias'], color='orange', linestyle='-', linewidth=2, label=f'Bias = {metrics["bias"]:.3f} pH')
    plt.xlabel('Estimation Error (pH units)')
    plt.ylabel('Probability Density')
    plt.title(f'Estimation Error Distribution (Kurtosis = {metrics["kurtosis"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_error_distribution.png')
    plt.close()

    # Figure 3: Uncertainty per pH Level
    plt.figure(figsize=(7, 5), dpi=300)
    plt.bar(pH_true, pH_std, width=0.4, color='#e07a5f', alpha=0.7)
    plt.axhline(metrics['U_pH']/2, color='red', linestyle='--', label=f'$U_{{pH}}/2 = {metrics["U_pH"]/2:.3f}$')
    plt.xlabel('Target pH ($pH_{{true}}$)')
    plt.ylabel('Standard Deviation (pH units)')
    plt.title('Uncertainty Distribution Across pH Scale')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig3_uncertainty_per_ph_level.png')
    plt.close()

    # Figure 4: Performance Summary Panel
    plt.figure(figsize=(6, 5), dpi=300)
    plt.axis('off')
    metrics_text = f"""
HYDROLOGICS PERFORMANCE METRICS
(Original Protocol Monte Carlo Simulation)

Goodness of Fit (R^2):        {metrics['R2']:.4f}
Root Mean Square Error:      {metrics['RMSE']:.3f} pH
Mean Absolute Error:         {metrics['MAE']:.3f} pH
Systematic Bias:             {metrics['bias']:.3f} pH
Expanded Uncertainty (k=2):  +-{metrics['U_pH']:.3f} pH
95% Coverage Rate:           {metrics['coverage']:.2f}%
Excess Kurtosis:             {metrics['kurtosis']:.4f}

CALIBRATION SUMMARY
Estimated K_cal:             {K_cal_estimated:.4f} V
Standard Uncertainty u_K:    +-{u_K:.4f} V
True Physical K_cal:         PARAMS['K_cal_true']:.4f V
Offset Magnitude:            {abs(K_cal_estimated - PARAMS['K_cal_true'])*1000:.2f} mV
"""
    plt.text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig('fig4_performance_summary.png')
    plt.close()

    print("\nFour separate figures saved successfully:")
    print("  • fig1_ph_target_vs_estimated.png")
    print("  • fig2_error_distribution.png")
    print("  • fig3_uncertainty_per_ph_level.png")
    print("  • fig4_performance_summary.png")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    print("="*70)
    print("HYDROLOGICS PROTOCOL: Sensorless pH Monitoring")
    print("Monte Carlo Validation with Redox Interferences (Zero-Mean Ohmic Fluctuations)")
    print("="*70)
    
    print(f"\nPhysical Parameters:")
    print(f"  Temperature: {T_KELVIN} K (25°C)")
    print(f"  Nernstian Slope: {S_NERNST*1000:.3f} mV/pH")
    print(f"  Ru: {PARAMS['Ru']} Ω")
    print(f"  sigma_noise: {PARAMS['sigma_noise']*1000:.2f} mA")
    print(f"  sigma_drift: {PARAMS['delta_E_ref_std']*1000:.1f} mV")
    print(f"  True K_cal: {PARAMS['K_cal_true']:.4f} V")
    print(f"  Interference Prob: {PARAMS['interference_prob']*100:.0f}%")
    
    # PHASE 1: Calibration
    print("\n[PHASE 1] System Calibration...")
    buffer_pH = [2.0, 4.0, 7.0, 10.0, 12.0]
    K_cal_estimated, u_K = calibrate_system(buffer_pH, n_replicates=3)
    print(f"  True K_cal:       {PARAMS['K_cal_true']:.4f} V")
    print(f"  Estimated K_cal:  {K_cal_estimated:.4f} +- {u_K:.4f} V")
    print(f"  Offset:           {abs(K_cal_estimated - PARAMS['K_cal_true'])*1000:.2f} mV")
    
    # PHASE 2: Monte Carlo Simulation
    print("\n[PHASE 2] Monte Carlo Simulation...")
    results = run_monte_carlo(K_cal_estimated, u_K)
    
    # PHASE 3: Metrics Calculation
    print("\n[PHASE 3] Metrics Calculation...")
    metrics = calculate_metrics(results)
    
    # PHASE 4: Final Results Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  R^2 = {metrics['R2']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.3f} pH")
    print(f"  MAE = {metrics['MAE']:.3f} pH")
    print(f"  Bias = {metrics['bias']:.3f} pH")
    print(f"  U_pH (k=2) = +-{metrics['U_pH']:.3f} pH")
    print(f"  Coverage 95% = {metrics['coverage']:.2f}%")
    print(f"  Kurtosis = {metrics['kurtosis']:.4f}")
    print("="*70)
    
    # PHASE 5: Separate Figures Generation
    generate_separate_figures(results, metrics, K_cal_estimated, u_K)
    
    # PHASE 6: Save Results JSON
    output = {
        'framework': 'HydroLogics',
        'version': 'zero_mean_ohmic',
        'parameters': {
            'Ru_Ohm': PARAMS['Ru'],
            'sigma_noise_A': PARAMS['sigma_noise'],
            'delta_E_ref_std_V': PARAMS['delta_E_ref_std'],
            'K_cal_true_V': PARAMS['K_cal_true'],
            'S_NERNST_V_per_pH': S_NERNST,
            'interference_prob': PARAMS['interference_prob'],
            'interference_magnitude_V': PARAMS['interference_magnitude']
        },
        'calibration': {
            'K_cal_estimated_V': K_cal_estimated,
            'u_K_V': u_K
        },
        'metrics': {k: v for k, v in metrics.items() 
                   if k not in ['residuals', 'pH_true']},
        'results_per_pH': [
            {
                'pH_true': r['pH_true'],
                'pH_mean': r['pH_mean'],
                'pH_std': r['pH_std']
            }
            for r in results
        ]
    }
    
    with open('hydrologics_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to: hydrologics_results.json")


if __name__ == "__main__":
    main()
# Theoretical Framework for Sensorless pH Monitoring in Hydrometallurgical Systems: A Voltammetric Overpotential Approach for Edge Computing

**Publication Title:** *Theoretical Framework for Sensorless pH Monitoring in Hydrometallurgical Systems: A Voltammetric Overpotential Approach for Edge Computing*  
**DOI / Reference:** [https://doi.org/10.5281/zenodo.18906011](https://doi.org/10.5281/zenodo.18906011)  
**Authors:** Daniel Vivas¹*, José Sanchez², Yndira Brito¹, José Duque³, and Arnaldo Armado⁴  

¹ *Aragua Project Unit (UPA), Faculty of Health Sciences, University of Carabobo, Maracay, Venezuela*  
² *Juana Ramírez Territorial Polytechnic University of the Plains, Valle de la Pascua, Guárico, Venezuela*  
³ *Laboratory of Pharmacological Biochemistry, BIOMED, Faculty of Health Sciences, University of Carabobo, Maracay, Venezuela*  
⁴ *LIBSA / AMBIOQUIM, Department of Chemistry, Faculty of Science and Technology, University of Carabobo, Valencia, Venezuela*  

\* **Corresponding Author:** Daniel Vivas (`dvivas1@uc.edu.ve`)  

---

## Overview

**Hydro-Logics** is a scientific computing toolkit designed for indirect pH estimation and virtual sensing in hydrometallurgical systems. By leveraging the physical dependence of the Hydrogen Evolution Reaction (HER) onset potential on hydrogen ion activity ($a_{\mathrm{H}^+}$), this software implements sequential state estimation algorithms (Kalman filtering) and robust signal processing pipelines to infer solution pH from Linear Sweep Voltammetry (LSV) data without direct potentiometric sensor immersion.

### Key Capabilities

1. **Synthetic Simulation Engine (`synthetic` mode):** Performs Monte Carlo experiments to model Nernstian behavior, calibration intercept uncertainty, and observation noise (including non-Gaussian outlier interference).

2. **Feature Extraction Engine (`extract` mode):** Detects operational onset potentials ($V_{\mathrm{onset}}$) from raw LSV polarization curves using Savitzky–Golay digital filtering and robust adaptive thresholding.

3. **Real-Data Calibration & Inference (`real` mode):** Implements a matrix-matched calibration module coupled with a Joseph-formulated sequential state estimator for sample pH prediction and uncertainty propagation ($U_{95\%}$).

> **Methodological Note:** Hydro-Logics provides an analytical and sequential filtering pipeline for empirical electrochemical data. It does not solve full continuum Nernst-Planck/Poisson-Boltzmann PDE systems, nor does it replace physical experimental validation with synthetic generation.

## Technical Architecture & Modes of Operation

The core script `simulation.py` provides three primary operational workflows:

| Mode | Objective | Primary Input | Outputs | 
| ----- | ----- | ----- | ----- | 
| **`synthetic`** | Evaluates theoretical performance, observer stability, and uncertainty propagation under defined noise regimes. | Simulation parameters, random seed, trial counts. | Monte Carlo evaluations, summary CSV/JSON metrics, diagnostics figures. | 
| **`extract`** | Extracts operational HER onset potentials ($V_{\mathrm{onset}}$) from current-voltage sweeps. | CSV file containing decremental LSV sweeps. | `extracted_onsets.csv` containing identified onset potentials. | 
| **`real`** | Calibrates model on standard buffer data and estimates pH of unknown target samples. | CSV file with matrix-matched calibration and test data. | `predictions.csv`, `summary.json`, error metrics, diagnostic scatter plots. | 

## Physical and Mathematical Foundation

### Nernstian Onset Model

The operational onset potential $V_{\mathrm{onset}}$ for the cathodic reduction of protons is modeled as a linear function of solution pH and absolute temperature $T$:

$$
V_{\mathrm{onset}} = K_{\mathrm{cal}} - S(T)\,\mathrm{pH} + \epsilon
$$

where $S(T)$ represents the ideal Nernstian slope:

$$
S(T) = \frac{2.303 \, R \, T}{F}
$$

At $T = 298.15\text{ K}$, $S(T) \approx 0.05916\text{ V/pH unit}$. The parameter $K_{\mathrm{cal}}$ aggregates the reference electrode offset, overpotential barriers ($\eta_{\mathrm{HER}}$), residual uncompensated ohmic drop ($i R_u$), and detection criteria.

### Deterministic Point Inversion

For a given calibration intercept $K_{\mathrm{cal}}$, the baseline pH estimate is obtained via direct inversion:

$$
\widehat{\mathrm{pH}} = \frac{K_{\mathrm{cal}} - V_{\mathrm{onset}}}{S(T)}
$$

### Sequential State Estimation (Kalman Filter)

To process multiple consecutive measurements while preserving numerical stability and tracking uncertainty, Hydro-Logics employs a scalar Kalman filter with the Joseph-formulated covariance measurement update.

* **State Model:** $x = \mathrm{pH}$, $H = -S(T)$

* **A Priori Prediction:**
  

  $$
  P^- = P_k + Q
  $$

* **Innovation:**
  

  $$
  \nu_k = y_k - (K_{\mathrm{cal}} + H x^-)
  $$

* **Kalman Gain:**
  

  $$
  K_k = \frac{P^- H}{H^2 P^- + R_v}
  $$

* **A Posteriori State Update:**
  

  $$
  x^+ = x^- + K_k \nu_k
  $$

* **Joseph-Form Covariance Update:**
  

  $$
  P^+ = (1 - K_k H)^2 P^- + K_k^2 R_v
  $$

Where $R_v$ is the composite variance representing reference drift, residual $iR$ drop, measurement noise, and bounded non-Gaussian interference:

$$
R_v = \sigma_{\mathrm{ref}}^2 + \sigma_{\mathrm{iR}}^2 + \sigma_{\mathrm{meas}}^2 + p_{\mathrm{interf}} \left( \frac{a^2 + ab + b^2}{3} \right)
$$

Uncertainty bounds ($U_{95\%} \approx 2 u_{\mathrm{total}}$) account for both state estimation variance $P^+$ and calibration intercept uncertainty $u_{K_{\mathrm{cal}}}$:

$$
u_{\mathrm{total}} = \sqrt{P^+ + \left( \frac{u_{K_{\mathrm{cal}}}}{\bar{S}} \right)^2}
$$

## Installation and Quick Start

### Prerequisites

* **Python:** Version 3.10 or higher

* **Dependencies:** `numpy`, `scipy`, `matplotlib`

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/hydrologics.git
cd hydrologics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
python -m pip install numpy scipy matplotlib
```

### Execution Examples

#### 1. System Self-Test

Verify installation integrity and mathematical routines:

```bash
python simulation.py --self-test
```

#### 2. Synthetic Monte Carlo Validation

Run a 1,050-trial Monte Carlo benchmark ($50\text{ trials} \times 21\text{ pH levels}$):

```bash
python simulation.py --mode synthetic \
  --seed 42 \
  --trials 50 \
  --replicates 5 \
  --temperature-C 25 \
  --output-dir synthetic_results
```

#### 3. Real Sample Inference

Execute calibration and evaluation on empirical CSV datasets:

```bash
python simulation.py --mode real \
  --input examples/real_onsets.csv \
  --output-dir real_results
```

#### 4. LSV Feature Extraction

Extract onset potentials from raw linear sweep voltammetry curves:

```bash
python simulation.py --mode extract \
  --input examples/lsv_sweeps.csv \
  --baseline-points 15 \
  --threshold-sigma 3 \
  --consecutive 3 \
  --output-dir extracted_lsv_results
```

## Data Schemas

### Onset Input Format (`real` mode)

The dataset must be formatted as a standard comma-separated values (CSV) file using standard decimal points.

```csv
sample_id,role,pH_reference,Vonset_V,temperature_C,batch_id
buf2,calibration,2.00,-0.443,25,batch_A
buf2,calibration,2.00,-0.441,25,batch_A
buf4,calibration,4.00,-0.558,25,batch_A
buf7,calibration,7.00,-0.734,25,batch_A
buf10,calibration,10.00,-0.913,25,batch_A
buf12,calibration,12.00,-1.032,25,batch_A
sample_01,test,6.30,-0.697,25,batch_A
sample_01,test,6.30,-0.691,25,batch_A
```

*Note: A minimum of three distinct calibration pH buffers is required. Test entries with missing `pH_reference` values will yield valid state predictions, though performance evaluation metrics (RMSE, $R^2$) will be omitted.*

### Voltammetry Sweep Format (`extract` mode)

Decremental LSV sweeps (scanning from positive towards negative potentials) require ordered voltage points and background baseline points prior to reaction onset.

```csv
sweep_id,sample_id,role,pH_reference,voltage_V,current_A,temperature_C,batch_id
s001,buf2,calibration,2.00,0.000,-0.000001,25,batch_A
s001,buf2,calibration,2.00,-0.005,-0.000001,25,batch_A
...
s002,sample_01,test,6.30,0.000,-0.000001,25,batch_A
```

## Integration with Virtual Sensing Architectures

Hydro-Logics can serve as a primary physical observer within broader multi-sensor virtual sensing networks (e.g., operator-theoretic frameworks such as MIMONet or DeepONet):

```
                        +---------------------------------------+
                        |      Physical LSV / Sensor Data       |
                        +---------------------------------------+
                                            |
                                            v
                        +---------------------------------------+
                        |  Hydro-Logics Analytical Observer     |
                        |   (Nernst-Kalman State Estimator)     |
                        +---------------------------------------+
                                            |
                                            v
                        +---------------------------------------+
                        |     Physically Bounded Baseline       |
                        +---------------------------------------+
                                            |
                                            v
                        +---------------------------------------+
                        | Data-Driven Residual / Spatial Net    |
                        | (e.g., MIMONet, Gaussian Process)     |
                        +---------------------------------------+
                                            |
                                            v
                        +---------------------------------------+
                        | Multi-Dimensional pH / Field Estimate |
                        +---------------------------------------+
```

1. **Level A (Physical Baseline - Included):** Direct inversion ($V_{\mathrm{onset}} \rightarrow \mathrm{pH}$) backed by sequential Kalman filtering. Highly efficient, interpretable, and suitable for edge embedded deployment.

2. **Level B (Hybrid Multivariable Sensing):** Data-driven residual mapping combining $V_{\mathrm{onset}}$, temperature, conductivity, ORP, and background ionic species via regularized regression or Gaussian Processes ($\mathrm{pH}_{\mathrm{pred}} = \mathrm{pH}_{\mathrm{physical}} + f_{\mathrm{residual}}(\mathbf{x})$).

3. **Level C (Deep Neural Operators):** Integration with spatial operator architectures (e.g., DeepONet/MIMONet) when mapping continuous 3D concentration or pH fields ($\mathrm{pH}(x,y,z,t)$) within complex reactor geometries.

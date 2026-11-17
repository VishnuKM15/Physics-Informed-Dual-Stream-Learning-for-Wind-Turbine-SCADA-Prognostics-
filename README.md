# Physics-Informed Dual-Stream Learning for Wind Turbine SCADA Prognostics (DualStreamV5)

A modular, physics-guided deep learning framework for **cross-farm early fault detection and prognostic health monitoring** in multi-megawatt industrial wind turbines using uniform 10-minute SCADA telemetry.

Benchmarked on the open-access **CARE-To-Compare** fleet corpus (36 turbines across 3 onshore and offshore wind farms), this framework addresses key operational challenges in industrial condition monitoring: sensor heterogeneity, operational regime shifts, non-stationary thermal lag, and ghost-sensor error collapse under cross-farm transfer.

---

## 1. Executive Summary & Problem Formulation

Supervisory Control and Data Acquisition (SCADA) systems record 10-minute operating averages across commercial wind fleets. While SCADA telemetry is universally collected, building generalizable early fault detection models across different wind farms is severely hindered by four structural challenges:

1. **Decoupled Physical Dynamics**: Aerodynamic and kinematic variables (wind speed, rotor speed, blade pitch) respond within seconds to atmospheric turbulence, whereas drivetrain thermodynamics (gearbox oil and main bearing temperatures) exhibit hours of thermal inertia. Monolithic sequential architectures conflate these distinct timescales.
2. **Sensor Heterogeneity & Ghost-Sensor Collapse**: Fleet turbines manufactured by different OEMs frequently log divergent channel subsets. When a model trained on source farms encounters missing or newly introduced sensors on an unseen target farm, unpenalized reconstruction residuals surge by up to $540\times$, completely destroying zero-shot anomaly scoring.
3. **Severe Operational Regime Shifts**: Aerodynamic operating points shift drastically across geographic topographies (onshore coastal hills vs. high-turbulence offshore basins). Purely data-driven autoencoders overfit to farm-specific operating envelopes.
4. **Transient Weather Glitches & False Alarms**: Atmospheric gusts and curtailment commands induce sharp transient reconstruction spikes, triggering costly false maintenance dispatches under standard pointwise thresholding.

### The DualStreamV5 Solution

`DualStreamV5` addresses these challenges through a unified physics-guided representation:
- **Dual-Stream Bi-LSTM Encoders**: Separates high-frequency aerodynamic boundaries from high-inertia drivetrain thermodynamics into independent bidirectional temporal streams before latent bottleneck fusion.
- **Observed-State Forward-Euler Thermal ODE Constraint**: Embeds a continuous lumped-parameter thermodynamic energy balance into discrete loss space ($\Delta t = 10\text{ min}$), learning physically grounded heat dissipation rates ($\beta_{\text{step}} \in (0, 0.5)$) and physical drivetrain time constants ($\tau \approx 27\text{--}40\text{ hours}$) with guaranteed asymptotic numerical stability.
- **Universal 12-Channel Physical Ontology & Core-8 Projection (`zeropad8`)**: Maps heterogeneous SCADA tags into an invariant physical coordinate frame, isolating an 8-channel invariant core that eliminates ghost-sensor error surge on target turbines.
- **Causal Rolling-Minimum Unanimity Gating**: Implements a causal $K=24$ (4-hour) rolling-minimum persistence filter that mathematically enforces $K$-of-$K$ unanimous exceedance, suppressing transient false alarms while providing early fault detection up to 18 days prior to component breakdown.

---

## 2. System Architecture & Overall Flow

```
====================================================================================================================
                                      DUALSTREAMV5 END-TO-END PIPELINE FLOW
====================================================================================================================

 Raw SCADA Logs (10-min)             Universal Physical Ontology                     Dual-Stream Bi-LSTM Encoders
 [Farms A, B, C CSV Telemetry]      [Invariant Coordinate Mapping]                   [Timescale Decoupling]

  +----------------------+             +-----------------------------+                  +-------------------------+
  | Wind Speed (m/s)     |             | Channel 0: v / 12.0         |                  | Stream 1: Aerodynamic   |
  | Rotor Speed (rpm)    |             | Channel 1: omega / 16.0     |                  | Bi-LSTM (d=6, h=128x2)  |
  | Active Power (kW)    |  =======>   | Channel 2: P / 3000.0       |  ============>   | -> Linear Proj (64)     |
  | Blade Pitch (deg)    |             | Channel 3: theta / 30.0     |                  +------------+------------+
  | Ambient Temp (C)     |             | Channel 4: T_amb / 40.0     |                               |
  | Theoretical P_residual|             | Channel 5: P_aero - P_meas  |                               |
  +----------------------+             +-----------------------------+                               |
  | Gearbox Oil Temp (C) |             | Channel 6: Delta T_oil / 60 |                               v
  | Main Bearing Temp (C)|  =======>   | Channel 7: Delta T_brg / 60 |                  +------------+------------+
  | Stator Current (A)   |             | Channel 8: I / 2500.0       |  ============>   | Cross-Stream Fusion     |
  | Stator Current Mask  |             | Channel 9: Mask_current     |                  | 128 -> 256 -> 128 (z)   |
  | Hydraulic Press (bar)|             | Channel 10: p_hyd / 200.0   |                  +------------+------------+
  | Hydraulic Mask       |             | Channel 11: Mask_hydraulic  |                               |
  +----------------------+             +-----------------------------+                               |
                                                                                                     |
                                                                                                     v
                                                                                        +-------------------------+
                                                                                        | Sequence Decoder MLP    |
                                                                                        | 128 -> 256 -> 512 -> 432|
                                                                                        | (Reconstructed X_hat)   |
                                                                                        +------------+------------+
                                                                                                     |
  +--------------------------------------------------------------------------------------------------+
  |                                        MULTI-OBJECTIVE OPTIMIZATION
  v
  [1] Weighted Reconstruction Loss: L_recon = sum( w_k * (X_hat_k - X_k)^2 ), w_11=0 (unpenalized mask)
  [2] Discrete Euler Thermal ODE:   d(Delta T)/dt = alpha * P_pu - beta * Delta T  (tau = 10 min / beta_step)
  [3] Multi-Kernel MMD Alignment:   Unbiased RBF kernel domain matching over sigma^2 in {1, 5, 10}
  +--------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
                         +----------------------------------------------------+
                         | Causal Last-Hour Core-8 MAD Standardization        |
                         | z_{t, k} = clip( (e_{t,k} - mu_k) / sigma_k, 0, 20)|
                         +-------------------------+--------------------------+
                                                   |
                                                   v
                         +----------------------------------------------------+
                         | Causal K=24 (4-Hour) Rolling-Minimum Persistence   |
                         | S_rmin[t] = min_{0 <= j < 24} S[t - j]             |
                         | (K-of-K Unanimous Exceedance Gate)                 |
                         +-------------------------+--------------------------+
                                                   |
                       +---------------------------+---------------------------+
                       |                                                       |
                       v                                                       v
         [Early Supervisory Warning]                             [Critical Alarm & Diagnosis]
         S_rmin > 2.81 (Source Normal p95)                       S_rmin > 4.50 (Critical Threshold)
         Prognostic Earliness: 3 to 18 Days                      Subsystem Attribution: Mech / Aero / Elec
====================================================================================================================
```

### Mathematical Formulation

#### 1. Discrete Forward-Euler Thermodynamic ODE
Heat generation in drivetrains is governed by continuous lumped-parameter thermodynamics:
$$\frac{d\Delta T}{dt} = \alpha_{\text{cont}} P_{\text{pu}}(t) - \beta_{\text{cont}} \Delta T(t)$$
where $\Delta T(t) = T(t) - T_{\text{ambient}}(t)$ is thermal rise above ambient, and $P_{\text{pu}} \in [0, 1]$ is electrical power. Discretizing via forward Euler with sampling interval $\Delta t = 10\text{ min}$:
$$\Delta T[t+1] = (1 - \beta_{\text{step}}) \Delta T[t] + \alpha_{\text{step}} P_{\text{pu}}[t]$$
To guarantee non-oscillatory asymptotic thermal decay during gradient descent, we introduce a bounded parameterization:
$$\beta_{\text{step}} = \beta_{\max} \cdot \sigma(\tilde{\beta}), \quad \text{with } \beta_{\max} = 0.5$$
This ensures the discrete pole $\lambda = 1 - \beta_{\text{step}} \in (0.5, 1.0)$ remains strictly inside the unit circle, yielding stable continuous drivetrain time constants:
$$\tau = \frac{\Delta t}{\beta_{\text{step}}} = \frac{10}{60 \cdot \beta_{\text{step}}} \quad \text{[hours]}$$

#### 2. Causal Persistence Gating
To eliminate spurious gust alerts, raw composite scores $S_t = \frac{1}{8}\sum_{k=0}^7 z_{t, k}$ pass through a causal rolling-minimum filter over $K=24$ steps ($4\text{ hours}$):
$$S_{\text{rmin}}[t] = \min_{0 \le j \le K-1} S[t-j]$$
Threshold exceedance satisfies the exact mathematical equivalence:
$$S_{\text{rmin}}[t] > \tau_{\text{src}} \iff \forall j \in \{0, \dots, K-1\}, \; S[t-j] > \tau_{\text{src}}$$
This acts as a strict $K$-of-$K$ unanimous exceedance gate that clears instantaneously upon a single normal timestep.


## 3. Evaluation Protocols & Experimental Framework

The repository implements four rigorous evaluation protocols on the 36-turbine CARE-to-Compare dataset:

### Protocol 1: Strict Zero-Shot Cross-Farm Transfer (`scripts/zeroshot_strict.py`)
- **Training Scope**: Trained strictly on Source Farms A and B. Target Farm C is completely unseen during training.
- **Decision Calibration**: Thresholds are frozen strictly using source normal percentiles ($p_{95} = 2.81$).
- **The Ghost-Sensor Autopsy**: Target Farm C turbines record hydraulic accumulator pressure but omit generator stator current. In unconstrained models, target-exclusive auxiliary channels cause a $540\times$ MSE surge, flagging $100\%$ of normal steps as false alarms ($\text{CARE} = 0.0255$).
- **Resolution (`zeropad8`)**: Zero-padding auxiliary channels ($X_{\cdot, \cdot, 8:12} \equiv 0$) at test time restores strict zero-shot performance to **$\text{CARE} = 0.6168$**, **$\text{AUC} = 0.9108$**, and $100\%$ event recall ($29/29$ failures detected).

### Protocol 2: Transductive Domain Adaptation (`scripts/final_frozen_evaluation.py`)
- **Training Scope**: Trained on Source Farms A+B plus healthy normal operation from Farm C (strictly zero target fault labels).
- **Alignment**: Unbiased MK-MMD aligns latent distributions across farms.
- **Performance**: Combined with causal 4-hour rolling-minimum persistence, attains **$\text{CARE} = 0.6661$**, **$\text{AUC} = 0.9090$**, **$\text{Accuracy} = 95.64\%$**, point-wise false alarm rate $3.75\%$, and detects $28/29$ failures with **$7.80\text{ days}$ median lead time**.

### Protocol 3: Classical Baseline Comparison (`scripts/phase3_baselines.py`)
- **Isolation Forest**: Evaluated on flattened 36-step Core-8 windows ($36 \times 8 = 288$ features). Attains $\text{CARE} = 0.5460$, $\text{AUC} = 0.6630$.
- **One-Class SVM**: RBF-kernel boundary on Core-8 representations. Attains $\text{CARE} = 0.5680$, $\text{AUC} = 0.8722$.
- **Standard Sequence Autoencoder**: Dense MLP autoencoder without physical streams or ODE constraints. Attains $\text{CARE} = 0.6397$, $\text{AUC} = 0.8996$.

### Protocol 4: Robustness Stress Testing (`scripts/evaluate_robustness.py`)
- **Missing Telemetry Robustness**: Evaluated under 5%, 10%, and 20% random sensor dropout with sample-and-hold forward-fill. Performance remains stable ($\Delta \text{CARE} < 0.05\%$) with $100\%$ event recall ($58/58$).
- **Outlier Contamination**: Evaluated under $+3\sigma$ to $+5\sigma$ impulsive noise injections at 1%, 5%, and 10%. Minor sensor glitches (1%) are absorbed by the rolling-minimum gate ($\text{CARE} = 0.5878$, false alarm rate $27.87\%$).
- **Seasonal Operating Generalization**: Evaluated across astronomical seasons over 156,354 chronological steps: Winter ($\text{CARE} = 0.7327$, $16/16$ events), Spring ($\text{CARE} = 0.5404$, $11/13$ events), Summer ($\text{CARE} = 0.6143$, $17/17$ events), and Autumn ($\text{CARE} = 0.7030$, $21/21$ events).

---

## 4. Input Data Schema & Ingestion Guide

The pipeline ingests standard tabular SCADA time series logged at uniform 10-minute intervals. 



- **Frequency**: Continuous 10-minute uniform grid. Missing rows should be forward-filled (`sample-and-hold`).
- **Feature Matching**: The pipeline automatically detects sensor presence. If a turbine lacks stator current or hydraulic pressure, the corresponding mask channel ($x_9$ or $x_{11}$) is set to $0.0$, and the feature is zero-padded.
- **Benchmark Access**: The public CARE-To-Compare benchmark dataset is described in:
  > Gück, C., Roelofs, C., & Faulstich, S. (2024). *CARE to Compare: A real-world benchmark dataset for early fault detection in wind turbines*. **Data**, 9(12), 138.


## 5. Computational Footprint & Edge Profile

`DualStreamV5` is designed for ultra-low computational latency, enabling real-time edge execution directly on industrial IPCs or embedded ARM processors (e.g., Raspberry Pi 4/5):

| Component / Layer | Input $\to$ Output Shape | Parameters | MACs per Window |
| :--- | :---: | :---: | :---: |
| `aero_lstm` (Bi-LSTM) | $(36, 6) \to (36, 256)$ | 139,264 | 4,939,776 |
| `aero_fc` (Linear) | $256 \to 64$ | 16,448 | 16,384 |
| `therm_lstm` (Bi-LSTM) | $(36, 6) \to (36, 256)$ | 139,264 | 4,939,776 |
| `therm_fc` (Linear) | $256 \to 64$ | 16,448 | 16,384 |
| `fusion` (2-MLP) | $128 \to 256 \to 128$ | 65,920 | 65,536 |
| `decoder` (3-MLP) | $128 \to 256 \to 512 \to 432$ | 386,224 | 385,024 |
| ODE Physical Scalars | $(\tilde{\alpha}_{\text{oil}}, \tilde{\beta}_{\text{oil}}, \tilde{\alpha}_{\text{brg}}, \tilde{\beta}_{\text{brg}})$ | 4 | -- |
| **Total Model Footprint** | **20.73 MFLOPs / window** | **763,572 (2.91 MiB)** | **10,362,880** |

- **Streaming Latency (Batch=1)**: $12.36\text{ ms}$ on standard Intel x86 CPU; $0.99\text{ ms}$ on NVIDIA L4 GPU.
- **Batched Throughput (Batch=128)**: $873.5\text{ windows/sec}$ on CPU ($1.14\text{ ms/win}$); $96,848\text{ windows/sec}$ on L4 GPU.

---

## 6. Intellectual Property, Weights & Data Notice

To comply with licensing agreements and protect commercial turbine operational privacy:
- **No Binary Model Weights**: Pre-trained model checkpoint binaries (`.ckpt`, `.pt`, `.pth`) are excluded. All model architectures, mathematical equations, loss formulations, and training scripts are fully provided for reproducible training.
- **No Proprietary Raw SCADA Data**: Raw turbine CSV time-series from proprietary fleets are omitted. Users may benchmark the pipeline using the public CARE-To-Compare dataset or their own SCADA data following the format specified in `configs/dataset_schema.json`.
- **Pure-Software Pipeline**: All scripts operate entirely offline over uniform time-series logs; no proprietary edge hardware or physical testbed is required.




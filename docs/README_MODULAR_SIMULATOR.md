
# Thesis MVP Simulator - Modular Monte Carlo Architecture

## Overview

This project implements a probabilistic Monte Carlo simulator focused on modeling large-scale Blob/Object Storage ecosystems.

The simulator generates synthetic datasets representing:

- file inventories
- storage lifecycle
- transfer operations
- infrastructure saturation
- operational incidents
- retries
- metadata degradation
- probabilistic storage errors
- storage tiers
- operational costs

The objective is to create realistic datasets suitable for:

- statistical analysis
- probability modeling
- machine learning
- anomaly detection
- capacity planning
- storage optimization
- cost simulation
- architectural experimentation

---

# High Level Architecture

```text
config/
├── global/
│   ├── cost_config.json
│   ├── lifecycle_config.json
│   ├── error_config.json
│   ├── noise_config.json
│   ├── realism_config.json
│   ├── correlation_config.json
│   └── time_distribution_config.json
│
└── escenarios/
    └── <scenario>/
        └── simulation_config.json
```

---

# Modular Configuration Design

The simulator was refactored to support a decoupled architecture.

The original monolithic configuration was split into:

- global technical behaviors
- scenario-specific workloads

This enables:

- reusable simulation behaviors
- independent scenario definitions
- easier experimentation
- cleaner statistical modeling
- extensibility

---

# Configuration Responsibilities

## simulation_config.json

Defines business/scenario behavior.

### Responsibilities

- daily volume
- file type distributions
- file sizes
- arrival process
- transfer configuration
- capacity
- outliers
- daily charge
- base load profiles

### Key Concepts

```text
daily_charge -> base_load_profile -> sampled load
```

Example:

```json
"daily_charge": {
  "monday": "medium",
  "friday": "high"
}
```

```json
"base_load_profiles": {
  "medium": {
    "min": 0.26,
    "max": 0.55
  }
}
```

---

## time_distribution_config.json

Defines temporal execution behavior.

### Responsibilities

- hourly distributions
- temporal load
- error multipliers
- execution variability
- weekly load factor
- weekly error factor

### Important

This module no longer owns `base_load`.

Base load now comes from:

```text
simulation_config.daily_charge
simulation_config.base_load_profiles
```

---

## lifecycle_config.json

Defines document lifecycle behavior.

### Responsibilities

- days stored
- last access
- access patterns
- lifecycle movement
- policy delays
- manual overrides

### Constraint

```text
days_since_last_access <= days_stored
```

---

## cost_config.json

Defines storage cost logic.

### Responsibilities

- storage tiers
- cost rules
- operational costs
- discounts
- billing periods

### Tier Resolution

```text
days_since_last_access -> storage_tier
```

---

## error_config.json

Defines probabilistic error behavior.

### Responsibilities

- base error probability
- technical multipliers
- severity levels
- error families
- subtype hierarchy

### Important Constraint

```text
error_probability <= 0.95
```

---

## noise_config.json

Defines residual statistical noise.

### Responsibilities

- gaussian perturbations
- jitter
- residual noise
- stochastic variability

### Important

Noise must trigger recalculations for:

- transfer duration
- queue pressure
- congestion
- storage tier
- storage cost

---

## realism_config.json

Defines operational imperfections.

### Responsibilities

- retries
- incidents
- saturation
- metadata quality
- entity skew
- dynamic capacity
- storage policy noise

---

## correlation_config.json

Defines cross-variable dependencies.

### Responsibilities

- conditional degradation
- congestion relationships
- timeout amplification
- queue pressure effects

### Recommended Execution Order

```text
noise
-> realism
-> correlations
```

---

# Simulation Pipeline

The new simulation pipeline is:

```text
1. load configs
2. resolve simulation date
3. calculate selected day
4. generate base files
5. apply temporal distribution
6. apply lifecycle
7. calculate storage tier
8. calculate transfer
9. apply noise
10. apply realism
11. apply correlations
12. calculate error probability
13. assign errors
14. apply retries
15. apply storage policy noise
16. recalculate costs
17. export dataset
18. generate summary
```

---

# Dynamic Day Resolution

The simulator now calculates the execution day automatically.

Example:

```bash
--simulation-date 2026-05-17
```

Internally becomes:

```text
selected_day = sunday
```

---

# New CLI

```bash
python run_simulation.py \
  --simulation-config config/escenarios/base/simulation_config.json \
  --time-distribution config/global/time_distribution_config.json \
  --lifecycle-config config/global/lifecycle_config.json \
  --cost-config config/global/cost_config.json \
  --error-config config/global/error_config.json \
  --noise-config config/global/noise_config.json \
  --realism-config config/global/realism_config.json \
  --correlation-config config/global/correlation_config.json \
  --simulation-date 2026-05-17
```

---

# Compatibility Constraints

The simulator must guarantee:

```text
queue_pressure between 0 and 1
error_probability <= 0.95
days_since_last_access <= days_stored
```

Mandatory recalculations:

```text
transfer_duration_sec after speed changes
storage_cost after storage_policy_noise
```

---

# Final Summary

The simulator summary now includes:

- simulation_date
- selected_day
- scenario
- total_files
- error_rate
- retry_rate
- queue_pressure_stats
- tier_distribution
- operational_costs

---

# Relationship Between Simulator and Analytics

The simulator feeds the analytical pipeline.

```text
Simulator
    ↓
Synthetic Dataset
    ↓
Exploratory Analysis
    ↓
Statistical Modeling
    ↓
Machine Learning
    ↓
Optimization
```

---

# Statistical and Research Goals

The simulator enables experimentation with:

- Monte Carlo simulation
- probability distributions
- stochastic systems
- storage optimization
- operational research
- anomaly detection
- cost prediction
- infrastructure saturation
- queue systems
- reliability analysis

---

# Recommended Future Improvements

## Infrastructure

- Dockerized simulator
- Airflow orchestration
- distributed execution
- cloud-native deployment

## Modeling

- copulas
- Gaussian Mixture Models
- Bayesian models
- Hidden Markov Models
- temporal forecasting

## Data Engineering

- parquet export
- partitioned datasets
- Spark support
- distributed generation

---

# Conclusion

The modular architecture transforms the simulator into a scalable probabilistic experimentation platform.

The decoupled configuration model enables:

- reusable simulation behaviors
- independent scenario creation
- cleaner experimentation
- more realistic operational modeling
- improved maintainability
- easier statistical research

The project now behaves as a configurable synthetic infrastructure laboratory for storage ecosystems.

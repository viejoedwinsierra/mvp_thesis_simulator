# Thesis MVP Simulator

This repository contains a Minimum Viable Product (MVP) designed to support thesis development in:

- Probability and Statistics  
- Big Data processing  
- Data Engineering pipelines  
- Machine Learning (data-centric modeling)  

---

# 🎯 Objective

To simulate, observe, and analyze a synthetic document ecosystem that represents a real-world object storage system.

The system is designed to:

- Generate a structured dataset at file level  
- Introduce controlled error conditions  
- Model probabilistic behavior of document generation  
- Enable statistical analysis and machine learning workflows  
- Materialize physical files based on the simulated dataset  

---

# 🧠 Conceptual Model

The system models a daily universe of documents defined as:

$$
M_d = \text{maximum valid documents per day}
$$

$$
E_d = M_d \cdot p_{error}
$$

$$
C_d = M_d - E_d
$$

Where:

- $$\(C_d\)$$: valid unique documents  
- $$\(E_d\)$$: error universe  

---

## Error Distribution

The error universe is distributed hierarchically:

$$
P(error) = \sum P(family) \cdot P(subtype \mid family)
$$

Error families include:

- duplicity cases  
- orphan files  
- null metadata  
- blob/storage errors  

---

# 🏗️ System Architecture

The system is divided into two independent modules:

## 1. Simulator (Data Generation)

Responsible for:

- probabilistic modeling  
- dataset generation  
- error injection  
- metadata simulation  

Output:

```text
output/dataset/blob_inventory.csv

# Thesis MVP Simulator

This repository contains a Minimum Viable Product (MVP) designed to support thesis development in:

- Probability and Statistics  
- Big Data processing  
- Data Engineering pipelines  
- ML-adjacent modeling  

---

# 🎯 Objective

To simulate, observe, and analyze a synthetic document ecosystem that includes:

- valid unique files
- controlled error cases
- metadata sidecar files (JSON)
- probabilistic hierarchy of events
- inventory extraction and dataset generation for statistical analysis

---

# 🧠 Conceptual Model

The system models a daily universe of documents based on:

\[
M_d = \text{maximum valid documents per day}
\]

\[
E_d = M_d \cdot p_{error}
\]

\[
C_d = M_d - E_d
\]

Where:

- \(C_d\): valid unique documents  
- \(E_d\): error universe  

The error universe is distributed hierarchically into:

- duplicity cases  
- orphan files  
- null metadata  
- blob/storage errors  

---

# ⚙️ Repository Structure

```text
src/
  simulator/
    allocator.py
    case_definitions.py
    config_models.py
    content_factory.py
    file_generator.py
    metadata_writer.py
    orchestrator.py

  inventario/
    build_inventory_csv.py   ← inventory + dataset generator

  run_simulation.py

config/
  simulation_config.json

docs/
  mathematical_foundation.md
  git_workflow.md

tests/
output/
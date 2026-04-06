# Thesis MVP Simulator

This repository is an MVP for thesis presentation and course approval support in probability, statistics, big data, and ML-adjacent subjects.

## Goal

Generate a daily synthetic universe of:

- valid unique files,
- controlled error cases,
- sidecar metadata,
- a documented probabilistic hierarchy,
- a structure ready for Git-based evolution.

## Repository structure

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
  run_simulation.py
config/
  simulation_config.json
docs/
  README.md
  mathematical_foundation.md
  git_workflow.md
tests/
output/
```

## Execution

```bash
python src/run_simulation.py --config config/simulation_config.json
```

## MVP principle

The orchestrator composes independent modules. It does not centralize all business logic in a single monolithic function.

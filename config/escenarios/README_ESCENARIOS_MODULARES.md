# Escenarios ajustados a arquitectura modular

Este paquete reemplaza los `simulation_config*.json` de `config/escenarios/<escenario>/`.

Cambios aplicados:
- Se eliminaron del escenario los bloques globales: `lifecycle`, `storage_tier`, `cost_model`, `errors`, `error_families` y `hash`.
- Se agregaron `daily_charge` y `base_load_profiles` en cada escenario.
- Se eliminó `simulation.simulation_date`; ahora la fecha se pasa por CLI con `--simulation-date`.
- Se mantuvieron los bloques propios del escenario:
  - `simulation.max_valid_files_per_day`
  - `simulation.output_dir`
  - `simulation.seed`
  - `arrival_process`
  - `capacity`
  - `outliers`
  - `file_types`
  - `transfer`
  - `scenario`
- Se creó el archivo faltante para `11_huerfanos_json_pdf`.

Ejecución recomendada:

```bash
python -m src.run_fileconfig_scenarios --start-date 2026-05-17 --end-date 2026-05-17 --max-scenarios 1
```

O para un escenario específico:

```bash
python -m src.run_fileconfig_scenarios --start-date 2026-05-17 --end-date 2026-05-17 --scenario-filter 01_base_controlada_100
```

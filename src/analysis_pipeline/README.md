# Analisis por escenario

Este paquete actualiza el flujo de analisis para trabajar con datasets generados por escenario:

```text
output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv
```

## Archivos incluidos

- `run_analysis.py`
- `config.py`
- `data_loader.py`

## Copia sugerida

```powershell
copy .\run_analysis.py .\src\run_analysis.py
copy .\config.py .\src\analysis_pipeline\config.py
copy .\data_loader.py .\src\analysis_pipeline\data_loader.py
```

## Ejecucion por escenario

```powershell
python .\src\run_analysis.py
```

## Ejecutar solo un escenario

```powershell
python .\src\run_analysis.py --scenario-filter red_lenta
```

## Ejecutar solo los primeros 2 escenarios

```powershell
python .\src\run_analysis.py --max-scenarios 2
```

## Incluir reporte global

```powershell
python .\src\run_analysis.py --include-global
```

## Salida esperada

```text
output/analysis/
  index_reportes_escenarios.html
  manifest_analysis_by_scenario.json
  01_base_controlada_100/
    html/report_01_base_controlada_100.html
    plots/descriptive/*.png
    tables/*.csv
```

## Objetivo

- Reporte HTML personalizado por escenario.
- Imagenes exportadas por escenario.
- Tablas CSV por escenario.
- Validacion de filas generadas vs max_valid_files_per_day.
- Lectura de configuracion original y runtime config.
- Comparacion de distribuciones observadas.

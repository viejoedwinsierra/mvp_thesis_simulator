#!/bin/bash

# ============================================
# MASTER RUN - SIMULACIÓN AUTOMÁTICA
# ============================================

CONFIG_DIR="./config"
TIME_CONFIG="./config/time_distribution_config.json"

echo "====================================="
echo "MASTER RUN SIMULACIÓN MONTE CARLO"
echo "====================================="

# Validación
if [ ! -f "$TIME_CONFIG" ]; then
    echo "❌ No existe time_distribution_config.json"
    exit 1
fi

# Buscar todos los configs de simulación
CONFIG_FILES=$(ls $CONFIG_DIR/simulation_config_*.json 2>/dev/null | sort)

if [ -z "$CONFIG_FILES" ]; then
    echo "❌ No se encontraron archivos simulation_config_*.json"
    exit 1
fi

echo ""
echo "📂 Configuraciones encontradas:"
echo "$CONFIG_FILES"
echo ""

# ============================================
# EJECUCIÓN SECUENCIAL (DÍA A DÍA)
# ============================================

for config_file in $CONFIG_FILES
do
    filename=$(basename "$config_file")

    echo "-------------------------------------"
    echo "▶ Ejecutando: $filename"
    echo "-------------------------------------"

    python -m src.run_simulation \
        --config "$config_file" \
        --time-distribution "$TIME_CONFIG"

    # Validar ejecución
    if [ $? -ne 0 ]; then
        echo "❌ ERROR en $filename"
        echo "⛔ Se detiene ejecución"
        exit 1
    fi

    echo "✔ Finalizado: $filename"
    echo ""

done

echo "====================================="
echo "✔ TODAS LAS SIMULACIONES COMPLETADAS"
echo "====================================="
# Mapa mental del simulador MVP de tesis

---

## 1. Visión general

El proyecto implementa un **simulador probabilístico de universos documentales diarios** para representar comportamientos de un sistema de almacenamiento tipo Blob / S3.

Su propósito es generar:

- archivos físicos simulados (`.pdf`)
- metadatos sidecar (`.json`)
- casos correctos y erróneos
- resumen diario consolidado

El flujo está organizado bajo una arquitectura modular donde cada archivo Python tiene una responsabilidad bien definida.

---

## 2. Estructura mental del sistema

```text
Simulación diaria
│
├── Configuración del universo
│   └── config_models.py
│
├── Definición de casos operativos
│   └── case_definitions.py
│
├── Distribución probabilística y asignación entera
│   └── allocator.py
│
├── Construcción del contenido lógico
│   └── content_factory.py
│
├── Materialización física de archivos
│   └── file_generator.py
│
├── Escritura de metadatos JSON
│   └── metadata_writer.py
│
└── Orquestación integral del proceso
    └── orchestrator.py
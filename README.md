# Thesis MVP Simulator

En las arquitecturas modernas con diferentes complejidades,se requieren datos altamente  descriptivos para interpretar las necesidades del negocio, los elementos de la informacion tales como volumen, velocidad, variedad, veracidad y valor asociados a la data generada por una empresa, es una buena fuente para modelar el negocio desde una perpectiva tecnica; para llegar a los modelos primero se debe analizar, enriquecer y describir la data generada por la operacion es la base fundamental para crear y entrenar  modelos capaces de describir y predicir los comportamientos de la infratestructura en funcion del negocio, como ejemplo la capacidad de computo y almacenamiento. Las empresas digitales cada ves mas populares, requieren de sistemas de almacenamiento donde tomar la desicion para antes para cualquier solucion de software era simple ahora es un tema importante porque los costos impactan directamente el rubro asociado a la produccion en entornos altamente digitales.
Una solucion altamente digital requiere etapas de diseñado y optimizacion antes de ser implementadas en ambientes producitivos, una forma de garantizar el buen diseño y funcionamiento de una solucion es el constante monitoreo pero genera costos en produccion que deberian ser medidos, para evitar el sobrecostos en ambientes productivos; Para este tipo de casos donde el entorno es altamente cambiante y con muchas variables, la simulacion es una de las soluciones mas importantes, donde se emplean funciones probabilisticas para describir el comportamiento de las variables del modelo y asi  garantizar el mejor modelo a partir de multiples simulaciones donde se podrian abordar diferentes casos de uso sin incurrir en gastos.
los datos pierden valor en funcion del tiempo y su uso durante el proceso de cualquier empresa y tomar desiciones sobre el costo de la infratestructura de procesamiento y almacenamiento;debe ser debe realizarse en bajo performance y costo para garantizar el ciclo de vida de la arquitecura, el procesamiento de informacion de poco valor debe hacerse e implementarse en arquitectura efimera tiempo de vida definido y garantizar autodestruccion de infratestructura en el pipeline reduce la superfice de exposicion donde hay menos maquinas encedidas son menos maquinas vulnerables.

---

# 🎯 Objective

To simulate, observe, and analyze a synthetic document ecosystem that represents a real-world object storage system.

The system is designed to:

- Generate a structured dataset at file level  
- Introduce controlled error conditions  
- Model probabilistic behavior of document generation  
- Enable statistical analysis and machine learning workflows  
- Materialize physical files based on the simulated dataset  

┌─────────────────────────────┐
│  1. Datos simulados / CSV    │
│  outputs Monte Carlo         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. run_analysis.py          │
│  Análisis de datos           │
└──────────────┬──────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────────┐  ┌────────────────┐
│ Descriptivo   │  │ Multivariado   │
│ univariado    │  │ exploratorio   │
└───────┬───────┘  └───────┬────────┘
        │                  │
        ▼                  ▼
┌───────────────┐  ┌────────────────┐
│ report_       │  │ report_        │
│ descriptive   │  │ advanced       │
│ .html         │  │ .html          │
└───────────────┘  └────────────────┘
        │                  │
        └────────┬─────────┘
                 ▼
┌─────────────────────────────┐
│  3. Evidencia exploratoria   │
│  relaciones, leakage,        │
│  outliers, transformaciones  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. run_modeling.py          │
│  Preparación + modelos       │
└──────────────┬──────────────┘
               │
      ┌────────┴─────────┐
      ▼                  ▼
┌────────────────┐  ┌────────────────────┐
│ Pre-modeling   │  │ Modeling final     │
│ datasets por   │  │ modelos            │
│ target         │  │ estadísticos       │
└───────┬────────┘  └────────┬───────────┘
        │                    │
        ▼                    ▼
┌────────────────┐  ┌────────────────────┐
│ report_        │  │ report_            │
│ premodeling    │  │ modeling           │
│ .html          │  │ .html              │
└────────────────┘  └────────────────────┘
        │                    │
        ▼                    ▼
┌────────────────────────────────────────┐
│  5. Evidencia final                     │
│  datasets preparados, métricas,         │
│  coeficientes, gráficas, reportes       │
└────────────────────────────────────────┘

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

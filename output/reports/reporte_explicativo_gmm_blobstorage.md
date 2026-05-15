# Reporte explicativo - Clustering GMM aplicado a BlobStorage

## 1. Objetivo del análisis

Este reporte analiza métricas simuladas u operativas de Azure Blob Storage para identificar patrones de comportamiento diario, agrupar días similares mediante clustering probabilístico y detectar posibles anomalías.

## 2. Flujo general del código

```text
CSV de entrada
  ↓
Validación y limpieza de datos
  ↓
Agrupación diaria de métricas
  ↓
Transformaciones logarítmicas
  ↓
Normalización con StandardScaler
  ↓
Selección de componentes con BIC
  ↓
Aplicación de Gaussian Mixture Model
  ↓
Asignación probabilística de clusters
  ↓
Visualizaciones mixtas: histogramas, gaussianas, PCA, heatmap y radar
```

## 3. Variables usadas en el modelo

### log_total_size_mb

- Variable original: `total_size_mb`
- Teoría aplicada: Transformación logarítmica
- Fórmula: `log(1 + total_size_mb)`
- Justificación: El tamaño total puede crecer mucho en días de alta carga. El logaritmo reduce el impacto de valores extremos y permite que el modelo compare mejor días normales contra días pesados.

### log_file_count

- Variable original: `file_count`
- Teoría aplicada: Transformación logarítmica
- Fórmula: `log(1 + file_count)`
- Justificación: La cantidad de archivos representa volumen operativo. Al aplicar logaritmo se evita que días con muchísimos archivos dominen completamente el clustering.

### avg_transfer_speed_mbps

- Variable original: `transfer_speed_mbps`
- Teoría aplicada: Promedio aritmético
- Fórmula: `sum(transfer_speed_mbps) / n`
- Justificación: Representa el rendimiento promedio de transferencia del BlobStorage. Ayuda a separar días rápidos de días lentos o congestionados.

### avg_queue_pressure

- Variable original: `queue_pressure`
- Teoría aplicada: Promedio aritmético
- Fórmula: `sum(queue_pressure) / n`
- Justificación: Representa la presión promedio de cola. Sirve para identificar congestión, acumulación de procesos o saturación.

### log_storage_cost

- Variable original: `total_storage_cost`
- Teoría aplicada: Transformación logarítmica
- Fórmula: `log(1 + total_storage_cost)`
- Justificación: El costo puede variar mucho entre días. El logaritmo estabiliza la escala y permite comparar costo con carga y rendimiento.

## 4. Teoría del modelo GMM

GMM significa Gaussian Mixture Model. Es un modelo de clustering probabilístico. A diferencia de K-Means, que asigna cada observación a un único grupo de forma rígida, GMM calcula la probabilidad de pertenencia de cada día a cada cluster.

La idea central es que los datos se explican como una mezcla de distribuciones gaussianas:

```text
P(x) = peso_1 * Gaussiana_1 + peso_2 * Gaussiana_2 + ... + peso_k * Gaussiana_k
```

Esto es útil para BlobStorage porque los días no siempre son completamente normales o completamente anómalos. Un día puede tener comportamiento mixto: carga media, costo alto, baja velocidad o presión de cola elevada.

## 5. Método del codo aplicado

El método del codo se aplica con K-Means como herramienta visual para analizar cuántos grupos naturales podrían existir. Aunque el modelo final usa GMM, el codo ayuda a explicar la segmentación.

### Resultados del método del codo

```text
 n_clusters   inertia  silhouette_score
          2 60.855975          0.537577
          3 43.576927          0.393409
          4 29.282470          0.414734
          5 21.420816          0.435733
          6 17.373324          0.360891
```

## 6. Selección del número de clusters con GMM

El mejor número de componentes seleccionado por BIC fue: **5**.

BIC penaliza modelos demasiado complejos. Por eso, no siempre gana el modelo con más clusters, sino el que logra mejor equilibrio entre ajuste y simplicidad.

### Resultados BIC/AIC

```text
 n_components        bic         aic
            2 114.071381   56.622289
            3  89.177252    2.303014
            4  36.824819  -79.474563
            5  20.665438 -125.059090
            6  46.108644 -129.041029
```

## 7. Interpretación automática de clusters

```text
 gmm_cluster  days  avg_total_size_mb  avg_file_count  avg_transfer_speed_mbps  avg_storage_cost  avg_queue_pressure  avg_likelihood  anomalies  low_confidence_days  avg_cluster_probability                                                                    interpretacion
           0     3      171573.290813     3669.000000                15.991802          5.623167            0.512548       15.802532          0                    0                      1.0                                              carga normal o baja, menor velocidad
           1    14      618292.472774    13275.857143                31.266498         20.095085            1.294073        2.970310          1                    0                      1.0                  alta carga, alto costo, mayor presión de cola, incluye anomalías
           2     8      578602.515287    12548.250000                19.468485         18.649096            1.931613        1.362850          1                    0                      1.0 alta carga, alto costo, menor velocidad, mayor presión de cola, incluye anomalías
           3     4      338544.557413     7235.250000                27.480754         11.107156            0.621964       10.022979          0                    0                      1.0                                                               carga normal o baja
           4     1      150351.329038     3455.000000                43.475566          5.141584            0.255733       26.542886          0                    0                      1.0                                                               carga normal o baja
```

## 8. Explicación de gráficos generados

### Método del codo aplicado con K-Means

Archivo generado: `kmeans_elbow_method.png`

![Método del codo aplicado con K-Means](kmeans_elbow_method.png)

**Por qué se seleccionó:** Justifica visualmente una cantidad razonable de clusters.

**Teoría aplicada:** La inercia mide compactación interna. El codo aparece cuando agregar clusters deja de aportar mucho.

**Cómo interpretarlo:** El punto donde la curva se aplana sugiere un número razonable de clusters.

### Evaluación Silhouette aplicada

Archivo generado: `kmeans_silhouette_method.png`

![Evaluación Silhouette aplicada](kmeans_silhouette_method.png)

**Por qué se seleccionó:** Complementa el codo midiendo separación entre grupos.

**Teoría aplicada:** Silhouette compara cercanía al propio cluster contra cercanía a otros clusters.

**Cómo interpretarlo:** Valores más altos indican clusters más separados y coherentes.

### Selección de componentes GMM usando BIC

Archivo generado: `gmm_bic_selection.png`

![Selección de componentes GMM usando BIC](gmm_bic_selection.png)

**Por qué se seleccionó:** Permite justificar el número final de componentes gaussianos.

**Teoría aplicada:** BIC equilibra ajuste del modelo y complejidad.

**Cómo interpretarlo:** El menor BIC indica el modelo más conveniente.

### Comparación mixta: size_mb original vs logarítmico

Archivo generado: `size_mb_original_log_mixed.png`

![Comparación mixta: size_mb original vs logarítmico](size_mb_original_log_mixed.png)

**Por qué se seleccionó:** Muestra por qué se usa logaritmo antes del clustering.

**Teoría aplicada:** El logaritmo comprime extremos y reduce sesgo positivo.

**Cómo interpretarlo:** La distribución logarítmica debe verse más compacta que la original.

### Componentes gaussianos GMM sobre log_total_size_mb

Archivo generado: `daily_log_total_size_gmm_components.png`

![Componentes gaussianos GMM sobre log_total_size_mb](daily_log_total_size_gmm_components.png)

**Por qué se seleccionó:** Es el gráfico más claro para explicar que GMM mezcla varias gaussianas.

**Teoría aplicada:** Cada curva es una gaussiana marginal del modelo para la variable log_total_size_mb.

**Cómo interpretarlo:** Curvas separadas sugieren patrones distintos de carga diaria.

### Visualización PCA con elipses gaussianas

Archivo generado: `gmm_pca_mixture_ellipses.png`

![Visualización PCA con elipses gaussianas](gmm_pca_mixture_ellipses.png)

**Por qué se seleccionó:** Reduce las variables del GMM a dos dimensiones y muestra la forma de los clusters.

**Teoría aplicada:** PCA proyecta las variables normalizadas y las elipses muestran dispersión gaussiana.

**Cómo interpretarlo:** Cada punto es un día; cada elipse representa la zona de mayor densidad de un componente GMM.

### Mapa de calor de probabilidades GMM

Archivo generado: `gmm_probability_heatmap.png`

![Mapa de calor de probabilidades GMM](gmm_probability_heatmap.png)

**Por qué se seleccionó:** Explica que GMM no asigna de forma rígida como K-Means.

**Teoría aplicada:** GMM calcula probabilidad de pertenencia a cada componente.

**Cómo interpretarlo:** Colores altos indican mayor probabilidad de pertenecer a un cluster.

### Perfil mixto promedio por cluster

Archivo generado: `gmm_cluster_profile_radar.png`

![Perfil mixto promedio por cluster](gmm_cluster_profile_radar.png)

**Por qué se seleccionó:** Resume varias variables en un solo gráfico interpretativo.

**Teoría aplicada:** Normaliza promedios de variables por cluster para comparar perfiles operativos.

**Cómo interpretarlo:** Cada línea representa un cluster y muestra su perfil de carga, costo, velocidad y presión.

### GMM - Tamaño total vs cantidad de archivos

Archivo generado: `gmm_scatter_size_count.png`

![GMM - Tamaño total vs cantidad de archivos](gmm_scatter_size_count.png)

**Por qué se seleccionó:** Muestra la relación entre volumen y número de archivos.

**Teoría aplicada:** Usa variables logarítmicas para comparar días de distinta escala.

**Cómo interpretarlo:** Cada punto es un día; color = cluster; tamaño = costo.

### Detección de anomalías GMM

Archivo generado: `gmm_anomalies.png`

![Detección de anomalías GMM](gmm_anomalies.png)

**Por qué se seleccionó:** Muestra días poco probables para el modelo.

**Teoría aplicada:** El log-likelihood bajo indica baja probabilidad bajo la mezcla gaussiana.

**Cómo interpretarlo:** Los puntos marcados como anomalía son los días menos probables.

## 9. Detección de anomalías

El código calcula `gmm_log_likelihood`, que representa qué tan probable es un día según el modelo aprendido. Los días con menor probabilidad son los más extraños.

Se usa el percentil 5 como umbral. Esto significa que el 5% de días menos probables se marcan como anomalías.

### Top 10 días más anómalos

```text
simulation_date  total_size_mb  file_count  total_storage_cost  avg_queue_pressure  gmm_cluster  gmm_log_likelihood  max_cluster_probability
     2026-04-10  757893.991913       16401           24.189258            1.508434            1            0.240983                      1.0
     2026-04-16  696491.076467       15175           22.469979            2.025863            2            0.552716                      1.0
     2026-04-29  859843.974613       15636           27.727608            1.490920            1            0.568570                      1.0
     2026-04-06  506953.227185       12030           15.635386            1.952481            2            0.580953                      1.0
     2026-04-07  491330.032194       11271           15.926277            1.037301            1            1.075761                      1.0
     2026-04-01  623140.615736       13855           20.272100            2.155306            2            1.094904                      1.0
     2026-04-22  639491.932144       14170           20.714283            2.196399            2            1.142151                      1.0
     2026-04-02  648272.411432       12685           21.132874            2.165593            2            1.228969                      1.0
     2026-04-13  494278.721129       11028           16.081337            0.873795            1            1.514086                      1.0
     2026-04-27  473042.774712       10224           15.375866            1.390809            2            1.887290                      1.0
```

## 10. Conclusión para exposición

Este código convierte datos operativos de BlobStorage en información analítica. Primero consolida CSV, limpia datos, construye métricas diarias, aplica transformaciones logarítmicas y normalización, selecciona componentes mediante BIC y aplica GMM para encontrar patrones de comportamiento.

Los nuevos gráficos mixtos permiten entender mejor el modelo: muestran la transformación de `size_mb`, las gaussianas que componen el GMM, la separación PCA, la probabilidad de pertenencia a cada cluster y el perfil operativo de cada grupo.


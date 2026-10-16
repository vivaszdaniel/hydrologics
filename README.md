# Hydro-Logics — simulador corregido, sensores virtuales y datos reales

**Fecha de revisión:** 16 de octubre de 2026  
**Revisión técnica:** José Alfredo Sánchez  
**Código:** [`simulation.py`]

> **Conclusión central:** sí se pueden construir **sensores virtuales de pH**, y sí hay **datos públicos descargables**. Sin embargo, las bases encontradas no proporcionan, de forma verificada, el conjunto completo y específico **LSV de evolución de hidrógeno (HER) + pH de referencia + múltiples matrices hidrometalúrgicas + temperatura y electrodo** necesario para validar directamente todo Hydro-Logics. Hay que combinar estudios de factibilidad con un conjunto local de calibración y de prueba. Este software **no convierte** datos sintéticos en evidencia experimental.

## 1. Contenido y alcance

El archivo `simulation.py` es **ejecutable** y tiene tres modos:

| Modo | Objetivo | Entrada | Salida |
|---|---|---|---|
| `synthetic` | Probar inversión de potencial de inicio y su incertidumbre | Parámetros de simulación | 1.050 evaluaciones (por defecto), CSV, JSON, 4 gráficos |
| `extract` | Extraer potencial de inicio operacional de curvas LSV | CSV con puntos corriente–voltaje por barrido | `onsets_extraidos.csv` |
| `real` | Calibrar y estimar pH de muestras experimentales | CSV de potenciales de inicio, con filas `calibration`/`test` | `predicciones.csv`, `resultados.json`, gráficos si hay etiquetas |

**Lo que no hace:** no resuelve Butler–Volmer/Poisson–Nernst–Planck, no genera curvas LSV sintéticas completas, no identifica químicamente HER frente a cualquier reacción catódica, no implementa MIMONet ni aprendizaje automático; tampoco ha sido validado con una celda física. El modo `extract` es un detector *operacional* de corriente por umbral.

## 2. Instalación y primer uso

Requisitos: **Python 3.10+**, `numpy`, `scipy`, `matplotlib`.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy scipy matplotlib

python simulation.py --self-test
python simulation.py --mode synthetic --output-dir resultados_sinteticos
```

Para una prueba rápida sin figuras:

```bash
python simulation.py --mode synthetic --trials 10 --replicates 5 --no-plots
```

Para consultar todas las opciones: `python simulation.py --help`.

**Salida por defecto:** carpeta `hydrologics_output/`, con `predicciones.csv`, `resultados.json` y hasta cuatro figuras `.png`. El nombre/carpeta puede modificarse con `--output-dir`. Usar `--seed` fija la semilla del generador NumPy, pero el resultado obtenido **no debe forzarse** a coincidir con las métricas del artículo original, ya que el diseño estadístico cambió.

## 3. Qué problemas había en el código entregado y cómo se corrigieron

1. **Fuga circular de calibración:** el original obtenía `K_cal_estimated` y luego **usaba ese mismo valor para generar los potenciales de evaluación**. Ahora los potenciales de evaluación se producen con un `K_cal_true` fijo, independiente del intercepto estimado durante la calibración. De este modo, el error de calibración afecta las predicciones y aparece en los resultados.
2. **Diseño Monte Carlo y dependencia temporal:** el original contabilizaba como 1.050 evaluaciones las 50 actualizaciones sucesivas de un filtro por pH, aunque sus estados intermedios incluían transitorios desde pH 7. Ahora se hacen **50 trayectorias de prueba por cada uno de 21 niveles**. Cada trayectoria reinicia el observador y contiene 5 lecturas por defecto; **solo la estimación final** se evalúa. Esto da 1.050 predicciones evaluables y 5.250 potenciales de prueba. Se comparte una única calibración entre las trayectorias (no son independientes en relación con el error común de calibración).
3. **Signo del filtro:** el **script original ya lo calculaba bien**, porque definía `H = -S_NERNST` y usaba `K = P H/(H²P + R)`. Sin embargo, **el pseudocódigo del artículo adjunto** emplea un signo positivo al calcular la ganancia/actualizar covarianza, inconsistente con `V = Kcal - S·pH`. El nuevo script conserva el signo correcto, verifica la monotonía y usa actualización de covarianza tipo **Joseph**.
4. **Separación de ruido y calibración:** la versión anterior añadía `uK²` a la varianza de cada medición, como si la calibración se repitiera e independizara en cada observación. Ahora el filtro utiliza la variabilidad por lectura; la incertidumbre del intercepto se agrega **una sola vez** al final como contribución común. El ajuste sigue siendo una aproximación y no sustituye un modelo de covarianzas completo.
5. **Medición de corriente vs voltaje:** `sigma_noise` del original estaba en amperios, pero el generador añadía 2 mV de ruido a un **potencial**; `Ru=15 Ω` estaba declarado pero **no participaba** en los cálculos. Se sustituyeron por tres desviaciones estándar **en voltios**: pseudo-referencia, caída óhmica residual y lectura de potencial. El efecto `i·Ru` físico debe medirse/modelarse con **corriente real y resistencia no compensada real**.
6. **Incertidumbre 95 %:** el cálculo original `U=2*std(errores)` extraía el intervalo de los mismos errores usados para evaluar cobertura; esa cobertura era **parcialmente auto-referente**. Ahora se calcula primero `U≈2u` a partir del filtro y de la calibración, **sin consultar el pH verdadero del conjunto de prueba**; después se mide cobertura observada. Bajo interferencias con colas pesadas, **±2u no garantiza un 95 %**.
7. **Calibración y datos reales:** se usan buffers de pH distintos (2, 4, 7, 10, 12) con 3 repeticiones, se promedian primero por buffer y se impide que un `sample_id` aparezca simultáneamente en calibración y prueba. Existe evaluación de CSV con y sin pH de referencia en la fase de prueba.
8. **Resultados/figuras:** se corrigieron textos literales sin interpolar del panel original, se evita llamar *intervalo de confianza 95 %* a un cálculo no calibrado y se distingue `R²/NSE` (idéntica fórmula en este caso, no dos verificaciones independientes).

**Importante:** estos cambios crean un experimento de simulación **más consistente**, no prueban que la relación empírica valga en todas las superficies, corrientes, matrices o intensidades iónicas. Por tanto, diferencias con `R²=0,9979`, `RMSE=0,138` u otros números publicados **no significan automáticamente un fallo**: se evalúa un diseño metodológico diferente.

## 4. Fundamento físico y matemático

El modelo utilizado para **potencial de inicio operacional** es

\[
V_{\mathrm{onset}} = K_{\mathrm{cal}} - S(T)\,\mathrm{pH} + \epsilon,
\qquad S(T) = \frac{2.303RT}{F},
\]

con `S(25 °C) ≈ 0,059159 V/pH`. La inversión elemental es

\[
\widehat{\mathrm{pH}}=\frac{K_{\mathrm{cal}}-V_{\mathrm{onset}}}{S(T)}.
\]

La identidad Nernstiana se deriva del **potencial de equilibrio** `H⁺/H₂`; el potencial de inicio observado **no es ese potencial reversible**: incorpora barreras de transferencia de carga, polarización de superficie, caída óhmica, electrodo de referencia y criterio de detección. Por ello la pendiente real puede **diferir de** `S(T)` y se requiere calibración *matrix-matched*. **No** debe aplicarse una segunda corrección de actividad si el pH de referencia ya representa actividad de H⁺ y el ajuste fue hecho con estándares comparables: así se evita contar dos veces un efecto.

El estimador secuencial usa `x=pH`, `H=-S(T)` y,

\[
P^- = P+Q,\quad \nu=y-(K_{cal}+Hx^-),\quad
K=\frac{P^-H}{H^2P^-+R_v},\quad
x^+=x^-+K\nu.
\]

La covarianza posterior se implementa en forma Joseph:

\[
P^+ = (1-KH)^2P^-+K^2R_v.
\]

**Unidades:** `x` en pH; `y` y `Kcal` en V; `H` en V/pH; `R_v` en V²; `P` y `Q` en pH². `Q=0` por defecto porque las cinco lecturas dentro de cada trayectoria representan un pH **constante**. Para cambios reales de pH hace falta modelar la dinámica y el tiempo entre lecturas. La pseudo-referencia se ha modelado con un ruido **independiente por lectura**; la deriva lenta real es **temporalmente correlacionada** y no queda representada adecuadamente por esa hipótesis.

La dispersión sintética usa, por lectura, `σref=15 mV`, `σóhmica residual=15 mV`, `σlectura=2 mV` y un término esporádico de signo aleatorio y magnitud uniforme **20–30 mV**, con probabilidad por lectura de **5 %**. La varianza de mezcla se incluye analíticamente:

\[
R_v = \sigma_{ref}^2+\sigma_{iR}^2+\sigma_{lectura}^2
+ p\,\frac{a^2+ab+b^2}{3}.
\]

Las colas de error **no son gaussianas**; el filtro de Kalman estándar es el punto de partida, no un filtro robusto garantizado. La incertidumbre de calibración se añade aproximadamente como `u_total² = P + (uK/S_promedio)²`, y `U≈2u_total`. Para resultados publicables sería preferible evaluar cobertura mediante un conjunto de **calibración conformal independiente** y modelar deriva/covarianza por sesión.

## 5. Ejecutar la simulación sintética

```bash
python simulation_corregida.py --mode synthetic \
  --seed 42 --trials 50 --replicates 5 \
  --temperature-C 25 \
  --interference-probability 0.05 \
  --output-dir resultados_sinteticos
```

En la ejecución de comprobación realizada para esta revisión (Python con `numpy/scipy`, semilla 42, 50×21 trayectorias, 5 lecturas por trayectoria) se obtuvo:

| Indicador de demostración | Valor |
|---|---:|
| Pendiente Nernst 25 °C | 59,159 mV/pH |
| Intercepto calibrado `Kcal` | −0,31819 V |
| Incertidumbre estándar de `Kcal` | 0,00360 V |
| Predicciones evaluadas | 1.050 |
| RMSE | **0,165 pH** |
| MAE | 0,132 pH |
| Sesgo | +0,0217 pH |
| R² | 0,9970 |
| Cobertura observada de `±2u` | 96,76 % |

**Todos estos valores son sintéticos, no son un resultado experimental.** Una buena cobertura en datos generados con las mismas hipótesis del observador puede reflejar el propio modelo y no demuestra que sea válida para una planta hidrometalúrgica.

Opciones útiles para escenarios alternativos:

```bash
# Quitar interferencias
python simulation_corregida.py --interference-probability 0

# Ajustar niveles de ruido a un experimento real (en mV)
python simulation_corregida.py --sigma-ref-mV 10 \
  --sigma-ohmic-mV 6 --sigma-measurement-mV 1

# Q de proceso mayor que cero para permitir movimiento del pH
python simulation_corregida.py --process-variance-ph2 0.0001
```

No se introdujo una opción de `Ru` en ohmios porque el generador no simula corriente: asignar `15 Ω` sin corriente `i` no basta para construir el término `iRu`.

## 6. Usar datos **reales**: CSV de onsets

Formato mínimo, coma como separador y punto decimal:

```csv
sample_id,role,pH_reference,Vonset_V,temperature_C,batch_id
buf2,calibration,2.00,-0.443,25,batch_A
buf2,calibration,2.00,-0.441,25,batch_A
buf2,calibration,2.00,-0.442,25,batch_A
buf4,calibration,4.00,-0.558,25,batch_A
buf7,calibration,7.00,-0.734,25,batch_A
buf10,calibration,10.00,-0.913,25,batch_A
buf12,calibration,12.00,-1.032,25,batch_A
prueba01,test,6.30,-0.697,25,batch_A
prueba01,test,6.30,-0.691,25,batch_A
```

**Estos números son solo ilustrativos, no datos observados.** Debe haber **al menos 3 pH de buffer distintos** de calibración; se recomiendan los cinco estándares, con **tres réplicas por buffer**, preferentemente iguales al medio de la muestra. Las muestras con `role=test` **nunca se utilizan** para estimar `Kcal`. Cuando no se disponga de `pH_reference` en las filas test, deje el campo vacío; el código predice, pero **no puede reportar RMSE, R² o cobertura**.

```bash
python simulation_corregida.py --mode real \
  --input onsets_reales.csv --output-dir resultados_reales
```

Cada `sample_id` de prueba define una serie de lecturas que se supone mantiene **pH constante**; por eso no debe agrupar varias etapas de pH variable bajo un mismo ID. `batch_id` opcional protege contra mezcla inadvertida de electrodos distintos si se rellenó consistentemente. El código rechaza múltiples lotes, pero **no detecta automáticamente** diferencias de formulación química, electrodo de referencia, tasa de barrido, laboratorio o matriz: registre estas variables y haga calibraciones separadas.

En datos reales, los errores estándar predeterminados (15, 15, 2 mV) son **hipótesis**, no mediciones instrumentales. Estímelos mediante réplicas, controles de deriva, resistencias medidas y controles de calidad. Revise que todos los potenciales estén en **la misma escala de referencia**: Ag/AgCl, SHE y RHE **no son intercambiables** sin conversión y metadatos. Una serie vs RHE puede ocultar parte de la dependencia Nernstiana buscada si no se transforma a una escala consistente.

## 7. Extraer onsets de curvas LSV

Puede crear `onsets_reales.csv` a partir de puntos LSV *decrementales* (de potencial más positivo a más negativo), sin confundirlos con las mediciones sintéticas:

```csv
sweep_id,sample_id,role,pH_reference,voltage_V,current_A,temperature_C,batch_id
s001,buf2,calibration,2.00,0.000,-0.000001,25,batch_A
s001,buf2,calibration,2.00,-0.005,-0.000001,25,batch_A
...
s002,prueba01,test,6.30,0.000,-0.000001,25,batch_A
```

**No es un archivo completo**: cada curva requiere decenas/cientos de puntos ordenados en **voltaje estrictamente decreciente**, con **al menos 15 puntos iniciales** de corriente de fondo y suficientes puntos posteriores.

```bash
python simulation_corregida.py --mode extract \
  --input barridos_LSV.csv \
  --baseline-points 15 --threshold-sigma 3 \
  --consecutive 3 --current-sign negative \
  --output-dir salidas_lsv

python simulation_corregida.py --mode real \
  --input salidas_lsv/onsets_extraidos.csv \
  --output-dir resultados_lsv_reales
```

Método de extracción: suavizado Savitzky–Golay de cinco puntos, línea de base con los primeros N puntos, escala de dispersión robusta complementada con desviación estándar, y primer cruce de `3σ` durante tres puntos consecutivos. `--current-sign positive` permite instrumentos donde la corriente catódica sea positiva. Se **descartan** curvas sin cruce; el programa **no extrapola** un onset inexistente.

**Advertencias de alta prioridad:**

- Un aumento de corriente puede deberse a **Fe/Cu, O₂ disuelto, otras reacciones redox, adsorción o capacitancia**: el cruce no certifica HER. Verificar con blancos, curvas hacia/desde voltajes de control, gas generado, materiales de electrodo y, cuando corresponda, voltametría/impedancia complementaria.
- Savitzky–Golay y `3σ` **no** establecen por sí solos trazabilidad metrológica ni el nivel 95 %; el detector debe estudiarse frente a barrido, umbral, ruido, fondo y corriente capacitiva.
- Si la corriente de fondo ya incluye HER, los primeros puntos **no** son una línea de base válida; cambie la ventana experimental (no solo el argumento de Python).
- Antes de comparar series, fije velocidad de barrido, paso, electrodos, geometría, área, temperatura, fuerza iónica y presión parcial de H₂. Es recomendable guardar curvas brutas y sus metadatos, no únicamente `Vonset`.

## 8. Pregunta 1 — ¿Se pueden usar sensores virtuales como en *Nature Communications*?

**Sí, pero hay que distinguir dos problemas diferentes.** El artículo de Kobayashi y colaboradores (2026) propone **MIMONet**, un **operador neuronal multi-entrada/multi-salida** que reconstruye *campos espaciales* inaccesibles usando sensores dispersos, condiciones operativas y coordenadas de consulta. Los ejemplos reales incluyen un campo de densidad de corriente de celda PEM, velocidad de viento a distintas alturas y estados oceánicos. **El artículo no publica un modelo entrenado que prediga pH desde HER**. Su validación PEM es sobre densidad de corriente, no pH en medios hidrometalúrgicos.

### Diseño recomendado para Hydro-Logics

**Nivel A — observador físico (este script):** `V_onset + temperatura + K_cal → pH`, con incertidumbre. Es ligero, interpretable y se puede ejecutar en un microcontrolador o computador perimetral. Mantenerlo como **línea base obligatoria**.

**Nivel B — sensor virtual escalar multivariable:** aprender una función `pH = f(V_onset, corriente/rasgos LSV, temperatura, conductividad, ORP, intensidad iónica estimada, química Fe/Cu, identidad electrodo, edad/calibración, velocidad de barrido)`. Requiere un conjunto real de entrenamiento con **pH de referencia** (electrodo trazable o análisis validado) y mediciones sincronizadas. Alternativas realistas con pocos datos: regresión regularizada, regresión de procesos gaussianos, random forest o gradient boosting; compararlas con el observador físico. Si hay metadatos suficientes, una regresión de *residuos* puede mejorar la interpretabilidad: `pH_pred = pH_físico + f_residual(variables)`.

**Nivel C — operador neuronal tipo MIMONet/DeepONet:** conveniente si se pretende reconstruir **pH(x,y,z,t)**, gradientes o múltiples campos acoplados dentro de un reactor con entradas distribuidas. Entradas: historias de corrientes/tensiones accesibles, flujo, temperatura, conductividad, composición y posición de sensores; salida: pH en una coordenada consultada. Necesita **campos de entrenamiento** de alta calidad, procedentes de simulaciones multiphysics contrastadas o instrumentación durante ensayos (aunque durante despliegue los puntos interiores sean inaccesibles). Si solo interesa **un escalar pH**, implementar inmediatamente MIMONet probablemente aumenta costo y riesgo de sobreajuste sin una ventaja demostrada.

**Punto crítico extraído del artículo:** parte de las evaluaciones de MIMONet consume una historia completa de entrada para reconstruir un estado final; eso se parece a un **suavizador de intervalo fijo**, no necesariamente a un predictor *causal* que opera sin ver el futuro. Para el despliegue verdaderamente en línea, restringir estrictamente las entradas a tiempos `≤ t` y medir latencia **total** (captura de LSV + procesamiento + inferencia), no solo de la red.

**Incertidumbre de IA:** MIMONet evalúa MC-dropout y calibración conformal para intervalos; una arquitectura neuronal **no** garantiza cobertura por el mero hecho de entrenarse. Mantener subconjuntos independientes para entrenamiento, selección de hiperparámetros, calibración conformal y prueba final, idealmente separados **por campaña/fecha/lote/matriz**, no por filas aleatorias casi idénticas.

**Recursos directos del artículo:**

- Artículo: https://doi.org/10.1038/s41467-026-77463-7
- Repositorio de MIMONet: https://github.com/kkazuma19/MIMONet
- Archivo público Zenodo (v2; **aprox. 3,7 GB**, código y datos, restricciones de licencia para datos): https://zenodo.org/records/21986357
- La referencia anterior mencionada en el PDF a `10.5281/zenodo.21224752` existe pero el registro consultado muestra **archivos restringidos**; por ello es preferible el enlace público v2 **si continúa disponible**. Confirmar condiciones de licencia: código MIT y datos en general CC BY-NC 4.0, con condiciones adicionales para productos de terceros.

> **Respuesta corta:** los sensores virtuales son **compatibles** con Hydro-Logics. La secuencia aconsejada es **modelo físico calibrado → modelo híbrido con datos de pH reales → operador neuronal solamente si se justifica la reconstrucción espacial**. MIMONet no elimina la necesidad de verdad de referencia durante el entrenamiento.

## 9. Pregunta 2 — ¿Hay datos públicos descargables?

**Sí.** He identificado estos repositorios y comprobado sus descripciones públicas. Se ordenan según utilidad **para aspectos distintos del proyecto**. La existencia de un archivo descargable **no equivale** a verificar que tenga todas las columnas y metadatos necesarios para Hydro-Logics; descargue e inspeccione cada archivo antes de integrarlo.

| Fuente verificable | Contenido anunciado | Qué permite probar | Qué **no** permite asumir |
|---|---|---|---|
| **Zenodo, HER Ag–Pd–Pt–Ru, 2026** — https://doi.org/10.5281/zenodo.20793807 | ZIP `SECCM_HER_activity.zip` con barridos **LSV de HER** + composiciones EDX | Métodos de lectura, extracción/robustez del umbral, variabilidad por material | No consta un abanico de pH de referencia por barrido; **no valida** por sí solo pH indirecto |
| **Zenodo, Cu₂₋ₓS para HER, 2023** — https://doi.org/10.5281/zenodo.14236127 | Paquete con LSV, cronopotenciometría, impedancia y evolución del electrodo; algunas curvas en KHCO₃ | Fouling/envejecimiento, variación cinética y estabilidad del onset | No se documenta un mapa calibrable pH 2–12 vs onset |
| **echemdb** — https://github.com/echemdb/electrochemistry-data | Archivo abierto de datos electroquímicos digitalizados y originales, sobre todo voltamogramas CV con metadatos | Desarrollo de lectores y filtrado/normalización de potenciales | No todos los registros son HER, LSV o tienen pH conocido |
| **UPWATER / Besòs, Zenodo 2026** — https://zenodo.org/records/20931195 | **XLSX descargable** con pH, conductividad, temperatura y concentraciones de metales, entre ellas Fe y Cu | Modelo de pH **multivariable** alternativo; análisis de matrices/redox | No incluye voltamogramas LSV/HER; **no** mide `Vonset` |
| **CaltechDATA, 2025** — https://data.caltech.edu/records/qrkev-fyr45 | Datos pH–composición–potencial (pH 1 y 7) en archivo JSON/ZIP de experimentación materiales | Ingeniería de variables, pruebas exploratorias pH/condición electroquímica | No debe tratarse como conjunto universal de `HER onset` vs pH |
| **Nature/MIMONet (Zenodo v2)** — https://zenodo.org/records/21986357 | Datasets de simulación y de casos reales, **incluido campo de corriente en celda de combustible PEM**, con modelos y checkpoints | Reproducir metodología de sensores virtuales con señales reales; transferencia metodológica | **No** contiene demostración de pH desde onset HER; tamaño ≈3,7 GB; usos de datos sujetos a licencias |
| **Dataset original de la celda PEM** — idUS https://hdl.handle.net/11441/160750 | Medidas reales de distribución de densidad de corriente bajo ciclos de carga | Prueba de sensores virtuales de **densidad de corriente** | No usar como etiqueta de pH para Hydro-Logics |

**Evitar falsa disponibilidad:** el conjunto *Hydrochemical and Isotopic Monitoring Dataset...* (https://zenodo.org/records/19096845) describe pH, ORP y conductividad, **pero está bajo embargo hasta el 31 de diciembre de 2026**, según su página consultada en octubre de 2026: **no incluirlo como descarga inmediata**.

### Orden de adquisición recomendado

1. **Descargar primero HER-AgPdPtRu y Cu₂₋ₓS**, inspeccionar archivos y metadatos, convertir curvas a CSV. Sirven para comprobar la sensibilidad del algoritmo de **extracción de onset**, no la calidad del predictor de pH en todo el intervalo.
2. **Descargar UPWATER**, construir un sensor virtual *alternativo* basado en las variables que sí están en la base (temperatura, conductividad, concentraciones, etc.). Reportar explícitamente que mide otro problema, con distinta matriz y objetivo, y nunca presentar su RMSE como validación HER.
3. **No mezclar ciegamente bases de distintas técnicas/escala de electrodo**: la referencia electroquímica, materiales, pH, temperatura, composición y tasa de barrido alteran el onset. Si no hay potencial de inicio **y** pH de referencia en el **mismo experimento**, no es un par de entrenamiento válido para el modelo físico `pH=f(Vonset)`.
4. **Recoger un conjunto pequeño pero cuidadosamente diseñado en laboratorio** para cerrar la laguna específica del artículo: cinco pH de calibración + niveles intermedios realmente independientes, repetidos sobre varios lotes de electrodo, matrices/impurezas, fechas y temperaturas, con lecturas pH de referencia y LSV sincronizadas.

## 10. Plan experimental mínimo propuesto para evidencia real

**No es una norma validada ni sustituye la evaluación de riesgos**; es un plan técnico para preparar publicación y poder comparar métodos sin fuga de información.

| Fase | Captura | Evaluación |
|---|---|---|
| **Piloto físico** | Soluciones buffer 2/4/7/10/12 en matriz compatible, cinco lotes de electrodos, ≥3 LSV por condición y pH de referencia paralelo | Ajustar `Kcal` por lote/matriz; evaluar pendiente medida frente a Nernst, sensibilidad al umbral, temperatura, corriente capacitiva y fallos de identificación HER |
| **Prueba hidrometalúrgica** | Soluciones de lixiviación/efluentes con Fe/Cu y distinto contenido iónico, mediciones de `Ru` o EIS, concentraciones y pH referencia | Comparar observador vs baseline constante, regresión multivariable y modelo híbrido; graficar residuales por metal, fuerza iónica, lote, fecha y electrodo |
| **Prueba prospectiva** | Campaña nueva con condiciones no usadas en entrenamiento | MAE, RMSE, sesgo, error p95, cobertura y amplitud de intervalo, porcentaje de barridos no detectados, latencia real y frecuencia de recalibración |

**Variables prioritarias a guardar por barrido:** `sample_id`, `sweep_id`, `timestamp`, `batch_id`, `electrode_material`, `reference_electrode`, `reference_scale`, `potential_V`, `current_A`, `temperature_C`, `pH_reference`, `pH_reference_method`, `sweep_rate_mV_s`, `potential_step_mV`, `ionic_strength_M`, `conductivity`, `Fe`, `Cu`, `Ru_ohm`, `matrix_id`, `operator_id`, `calibration_date`, `role`. No es necesario que todo entre en `simulation_corregida.py` hoy, pero **sí es importante conservarlo** en un registro maestro verificable.

Recomendación científica: validar los lotes y matrices difíciles por separado, y aplicar selección de modelo y calibración de intervalos **sin mirar el conjunto de prueba**. Documentar valores fuera del rango de calibración como **extrapolación**; prohibir su uso en aplicaciones clínicas hasta que exista validación y conformidad regulatoria específicas.

## 11. Interpretación de las métricas y errores frecuentes

- `RMSE`: error cuadrático medio, en unidades pH. No se interpreta como ±95 %.
- `MAE`: media de |error|, en pH.
- `bias`: `pH_estimado - pH_referencia`, firmado.
- `R²` y `NSE`: coinciden algebraicamente con la definición aplicada; son insensibles a la prueba de extrapolación y pueden ser grandes cuando el rango pH es muy amplio. Reportar siempre métricas por subrango y por matriz.
- `median_U95_approx_pH`: mediana de `2u` calculada **antes** de conocer la etiqueta en test. **Aproximación**, no certificado de 95 %.
- `empirical_coverage_pct`: fracción de etiquetas de prueba dentro de ese intervalo. Una cobertura del 95 % en el mismo modelo sintético **no garantiza** la misma cobertura experimental.
- `excess_kurtosis`: sensibilidad a colas y a muestras atípicas; reportar N y resguardar errores individuales.
- **Deriva:** el filtro escalar empleado presupone ruidos por lectura aproximadamente independientes. Un offset que se arrastra durante una sesión **no se reduce** promediando LSV. Para producción, añadir un estado de deriva `ΔE_ref` con restricción/medida auxiliar, calibración temporal o un filtro robusto bien especificado.
- **Identificabilidad:** con un único onset, pH y sobrepotencial pueden confundirse: distintas parejas `(pH, η)` producen el mismo voltaje. Esta es una limitación física, no un problema corregible agregando capas a una red neuronal.

## 12. Licencia, trazabilidad y fuentes

El `simulation.py` aportado declara **GNU GPL v3** y acredita D.V. y J.S. como autores del código original. `simulation_corregida.py` conserva esa atribución en su cabecera como obra derivada y se distribuye bajo **GNU GPL v3**. Antes de redistribuirlo junto a otros componentes, conservar la licencia y revisar los derechos aplicables. Los datos de terceros mantienen **sus propias licencias** y pueden prohibir uso comercial, como ocurre con parte del archivo de MIMONet.

**Documentos principales revisados:**

1. Vivas, D., Sánchez, J., Brito, Y., Duque, J. y Armado, A. *Theoretical Framework for Sensorless pH Monitoring in Hydrometallurgical Systems*, manuscrito aportado (2026).
2. Kobayashi, K. et al. (2026). *Virtual sensing to enable real-time monitoring of inaccessible locations & unmeasurable parameters*. *Nature Communications*. https://doi.org/10.1038/s41467-026-77463-7
3. Código asociado: https://github.com/kkazuma19/MIMONet y https://doi.org/10.5281/zenodo.21986357
4. Bases potencialmente reutilizables: https://doi.org/10.5281/zenodo.20793807; https://doi.org/10.5281/zenodo.14236127; https://github.com/echemdb/electrochemistry-data; https://zenodo.org/records/20931195; https://data.caltech.edu/records/qrkev-fyr45.

**Alcance de la verificación:** se examinó el artículo, su sección *Data availability*, las páginas de repositorios y descripciones de datasets; se **ejecutó el script corregido** con pruebas internas, simulación y CSV de ensayo creado para probar el código. **No se descargaron ni auditaron todos los datos originales de terceros ni se instrumentó un ensayo físico.** La calidad experimental y adecuación legal deben verificarse caso por caso.

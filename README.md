# _SINBA ResNet_: User Guide
Cruz, D. Emmanuel $^{\dagger *}$

</small>$^\dagger$ Facultad de Ciencias Físico Matemáticas, Benemérita Universidad Autónoma de Puebla.</small>  
</small>$^*$ emmanuel.cruzd@alumno.buap.mx</small>

### Abstract 

La clasificación de señal y ruido de eventos de altas energías implica hasta la fecha un reto para la física experimental moderna. Múltiples métodos de aprendizaje de máquina para su diferenciación han sido empleados como lo son los árboles de decisión, los Random Forest o el clustering con un buen desempeño y son el estado del arte más empleado. En este trabajo se propone utilizar un diseño de red neuronal artificial para la clasificación de estos eventos. *_SINBA_* (SIgnal aNd BAckground) es un programa que ayuda a la búsqueda de hiperparámetros para una red neuronal tipo ResNet, que clasifica eventos señal y ruido de eventos de colisiones de partículas simuladas con Delphes. El programa permite elegir la búsqueda de eventos en los que una partícula haya decaído a pares de partículas y reconstruir su masa invariante para crear una base de datos basada en el _Higgs Boson Challenge_ con la que es entrenada la red . Se reporta un ejemplo en el que se entrenó una red para la diferenciación de eventos que produjeron un Higgs que decae a un par leptónico $\ell^+ / \ell^-$ y a un quark b/anti-b $b / \bar{b}$ y un quark anti_s/b $s / \bar{s}$.

## Introducción

El programa SINBA[^1] tiene como principal objeto facilitar el diseño de una red neuronal tipo ResNet para realizar la diferenciación de eventos señal y eventos ruido de colisiones de partículas. Su diseño está enfocado a trabajar con archivos .ROOT derivados de simulaciones Delphes. 

El programa requiere la instalación de la paquetería Uproot, la cual es una alternativa para la paquetería PyRoot, y permite trabajar archivos .ROOT con librerías más familiares como Numpy y Pandas. La documentación completa de la paquetería Uproot puede encontrarse en [Uproot Library](https://uproot.readthedocs.io/en/stable/index.html).

SINBA permite el cálculo de variables para el entrenamiento de una red neuronal basándose en las recomendadas en el _Higgs Boson Challenge_ del CERN. Una descripción más detallada sobre cada una de estas variables puede encontrarse en [1]. 

Entre las configuraciones de los programas empleados hay anexada una opción para guardar los DataFrames
creados para su funcionamiento. Esto ha sido diseñado de esta manera para permitir al usuario guardar las bases de datos en el futuro y permitir entonces crear, por ejemplo archivos .csv, que pudieran resultar más cómodos para trabajar con métodos similares o diferentes de aprendizaje de máquina.

## Variables Calculator

El primer módulo esencial para SINBA es variables_calculator. Este programa permite realizar el cálculo de
cantidades importantes a utilizar para el entrenamiento de la red neuronal. Entre estas se encuentan la masa invariante, la masa transversa, la separación de la pseudo-velocidad, el momento transverso, entre otros.

Las variables usadas para el entrenamiento se dividen esencialmente en dos tipos: Las primarias y las derivadas. Como su nombre indica, las primarias son las que se obtienen de la información en crudo del archivo ROOT, e.g. el número de jets en el evento; por su lado, las variables derivadas serán aquellas que se obtienen a partir de cálculos con las primarias. La variables primarias son etiquetadas en la base de datos o DataFrame crado con el prefijo PRI. Similarmente, las variables derivadas son etiquetadas con un prefijo DER.

En las simulaciones Delphes puede permitirse el caso en el que en un evento hay más de una partícula de un tipo, e.g. que haya tres electrones, y en todo caso la elección de la variable será acorde a la correspondiente de la partícula que tenga el mayor valor de momento transverso[^2]. 

Las variables posibles a calcularse usando este módulo son: 

- DER_INV_m: Reconstrucción de la masa invariante de una partícula que decae a otras dos. Se calcula con la función `inv_m`:

```
def inv_m(file, p1, p2, charge = 0, flavor1 = 0, flavor2 = 0, save_df = False, graph = False, mass = 0)
```
- Variables de la función:
    - `File`. Archivo .ROOT con el que se trabajará.
    - `p1`, `p2`. Partícula 1 y 2 de la que se reconstuirá la masa invariante.
    - `charge`. Carga teórica de la partícula reconstruida.
    - `flavor1`. Flavor del primer jet. Configurado por defecto como 0 para que si `p1` no es un jet, no se considere entonces su flavor.
    - `flavor2`. Flavor del segundo jet. Configurado por defecto como 0 para que si `p2` no es un jet, no se considere entonces su flavor.
    - `save_df`. Opción de guardar el DataFrame creado como un archivo .csv.
    - `graph = False`. Para graficar la distribución de la masa invariante calculada en los eventos. Por defecto no se realiza.
    - `mass`. Para añadir una línea de referencia que indique un punto de la gráfica de la distribución de la masa invariante reconstruida por evento. Por defecto si `mass = 0` no se realiza el graficado.

En esta función, además, se calculan las variables:

- PRI_pt_1 y PRI_pt_2. Momentos transversos de la partícula 1 y 2.
- PRI_eta_1 y PRI_eta_2. Pseudorapidez de las partículas 1 y 2. 
- PRI_phi_1 y PRI_phi_2. Ángulos $\phi$ de las partículas 1 y 2. 

Las demás variables siguen una arquitectura similar:

- DER_pseudorapidity_separation. Valor absoluto de la separación de la pseudorapidez entre dos partículas A y B:

$$
|(\eta_A - \eta_b)|
$$

```
def pseudorapidity_separation(file, p1, p2, flavor1 = 0, flavor2 = 0, charge = 0, save_df = False)
```

- DER_trans_m. Masa transversa entre un par de partículas. 

$$
m_{\mathrm{tr}}(a, b) = \sqrt{\left( \sqrt{a_x^2 + a_y^2} + \sqrt{b_x^2 + b_y^2} \right)^2 - (a_x + b_x)^2 - (a_y + b_y)^2}
$$

```
def trans_m(file, p1, p2, charge = 0, flavor1 = 0, flavor2 = 0, save_df = False, graph = False, mass = 0)
```

- PRI_pt. Momento transverso, $p_t$ de una partícula:

$$
p_t = \sqrt{p_x^2 + p_y^2}
$$

```
def trans_momentum(file, p1, p2, charge, flavor1 = 0, flavor2 = 0, save_df = True)
```

- PRI_jet_leading[...] y PRI_jet_subleading[...]. $P_t$, $\eta$ y $\phi$ del jet con mayor momento transverso y $P_t$, $\eta$ y $\phi$ del segundo jet con mayor momento transverso de cada evento.

```
def leading_n_subleading_jets(file)
```

- PRI_met y PRI_met_phi. Energía perdida transversa y su ángulo $\phi$. Se calculan con la función `met()`:

```
def met(file)
```

- PRI_jet_all_pt. Número de jets totales en el evento. 
```
def PRI_jet_all_pt(file)
```
- DER_prodeta_jet_jet. Producto de las pseudorapidez entre dos jets. Sólo considérese cuando la masa invariante reconstruida implique dos jets.
```
def DER_prodeta_jet_jet(file, flavor1, flavor2, charge = 0, save_df = False)
```
- DER_deltaeta_jet_jet. Valor absoluto de la separación de la pseudorapidez entre dos jets. Se considera únicamente cuando la masa invariante reconstruida implique dos jets.
```
def DER_deltaeta_jet_jet(file, j1, j2, flavor1, flavor2, charge = 0, save_df = False)
```

## DataSet Creator

El módulo data_set_creator permite la creación y el guardado de bases de datos para los eventos señal y ruido respectivamente. Así mismo, da posibilidad de crear directamente los conjuntos de entrenamiento, validación y prueba con los que se puede entrenar un modelo de red neuronal (o cualquier otro método de aprendizaje de máquina deseado que lo permita). En este programa también ha sido definida una función que permite observar la matriz de correlación entre las variables que conforman a un conjunto de datos creado con la función dataset() de este programa. 

- `dataset()`. La función crea un DataFrame usando las variables definidas en variables_calculator. 

```
def dataset(file, p1, p2, flavor1 = 0, flavor2 = 0, charge = 0, Signal = False, Noise = False, save_csv = False)
```
Sus variables son: 

- `File`. Ruta del archivo .ROOT ha trabajar.
- `p1` y `p2`. Partículas 1 y dos con las que se reconstruirá la masa invariante.
- `lavor 1` y `flavor2`. Flavor del jet correspondiente a p1 y flavor del jet correspondiente a p2 por si es recorrido. Por defecto se les asigna el valor de 0 indicando que no se trabaja con jets pero el usuario puede hacerlo. 
- `charge`. Carga teórica de la partícula a la que se le reconstruye su masa invariante. 
- `Signal = False`, `Noise = False`. Cambiese el valor de `Signal` o `Noise` a `True` para asignar una etiqueta 1 en caso de que sean eventos señal o etiqueta 0 si son eventos ruido. No pueden ser ambas True al mismo tiempo. 
- `save_csv`. Para guardar el DataFrame como un .csv.

El DataFrame obtenido tiene como columnas:
- PRI_jet_all_pt
- PRI_jet_num
- PRI_jet_leading_pt
- PRI_jet_subleading_pt
- PRI_jet_leading_eta
- PRI_jet_subleading_eta
- PRI_jet_leading_phi
- PRI_jet_subleading_phi
- PRI_met
- PRI_met_phi
- Der_prodeta_jet_jet
- DER_deltaeta_jet_jet
- PRI_pt_1
- PRI_pt_2
- PRI_eta_1
- PRI_eta_2
- PRI_phi_1
- PRI_phi_2
- DER_INV_m
- label
- DER_mass_lep. Masa invariante de los pares leptónicos de cada evento.

Cada una de estas variables conforme a lo que se explicó en la sección anterior. 

- `model_sets()`. Este programa crea los conjuntos que se utilizarán para entrenar un modelo clasificador de red neuronal artificial (RNA) de señal y ruido.

    Por defecto, no se considera un subconjunto de prueba. Puede configurarlo para crearlo estableciendo `test = True`.

    El porcentaje para cada subconjunto está configurado por defecto de la siguiente manera:

    - 50 % de datos de entrenamiento

    - 50 % de datos de validación

    - 0 % de datos de prueba (ya que no se consideran)

    El usuario puede modificar estos valores.

    `signal_df` y `noise_df` corresponden a DataFrames de Pandas que el usuario debe crear utilizando la función `dataset()` de este programa.

    El valor de `esc` se utiliza para aumentar el tamaño del conjunto de señal, ya se ha observado que los conjuntos de ruido en ocasiones pueden ser muy pequeños y los modelos RNA ResNet pueden tener problemas para clasificar.

    Por defecto, `esc = 1`, pero el usuario puede modificarlo.

```
def model_sets(signal_df, noise_df, test = False, weight_classes = False, train_per = 0.5, val_per = 0.5, test_per = 0., esc = 1., random_state = 45)
```

La opción `weight_classes` es asignada en el caso que haya desbalanceo en los conjuntos señal y ruido. Hace una proporción que permite darle más importancia durante el entrenamiento al conjunto que tenga menor número de datos. 

- `corr()`. Función para ver la matriz de correlación entre las variables de un conjunto de datos creado en este programa. 
```
def corr(df, cmap = "coolwarm", method = "pearson", fig_size = (20,20), droplabel = True)
```
- `df`. DataFrame a analizar.
- `method`. Método a utilizar para analizar la correlación entre las variables.
- `droplabel = True`. Por defecto elimina la columna con la etiquita 0/1 del conjunto de datos.

## Residual Block

Módulo con una función que ayuda a crear el bloque residual utilizado en los experimentos con Optuna o el modelo final.

Permite al usuario configurar las unidades del perceptrón por capa residual, la función de activación para todo el
bloque residual, si se utiliza Dropout o no, la tasa de Dropout por capa (en caso afirmativo), elegir un regularizador de capa,
y el valor del regularizador. Tiene valores asignados por defecto por si el usuario desea emplear esta función en otro programa.

```
def residual_block(x, perceptrons, activation, dropout_rate = 0.0, reg_value = 1.0e-3, regularizer = "l2")
```

## Experiment

El módulo `experiment` permite al usuario realizar experimentos con Optuna y WandB para la búsqueda de hiperparámetros. Da la opción al usuario para qué eliga el caso que requiera. 

En el programa se utiliza Optuna para que sugiera el número de capas, las unidades de neurona por capa, la función de activación en todas las capas (por defecto se elije entre relu, relu6 y leaky_relu, pero se pueden modificar), el valor de la taza de aprendizaje ($\eta$), qué regularizador ocupar (L1/L2) y su valor, si ocupar Dropout en las capas residuales y el valor del Dropout en cada capa, el optimizador (Se seleccionaron por defecto SGD, Adam, RMSprop y AdamW) y si usar balanceo de clase.

- `objective()`. Para realizar experimentos sin seguimiento en Optuna.
```
def objective(trial, X_train, X_val, y_train, y_val, activation_f = ["relu","relu6","leaky_relu"],layers_interval = (15,20), units_interval = (128,256), regularizer_interval = (1e-7, 1e-5), opt = ["sgd", "adam", "rmsprop", "adamw"], eta_interval = (2.5e-4, 1e-3,), class_weight = {}, dropout_interval = (0.1, 0.15), custom_optimizer = False)
```

- `X_train`,`X_val`, `y_train`, `y_val` corresponden a los conjuntos de entrenamiento y validación del modelo. 
- `activation_f`permite la elección de funciones de activación que optuna eligirá para ocuparse en todas las capas de cada modelo.
- `layer_interval` permite la elección del intervalo de búsqueda de Optuna para el número de capas residuales.
- `units_interval` permite la elección del intervalo de búsqueda de Optuna para el número de neuronas por capa.
- `regularizer_interval` permite la elección del intervalo de búsqueda de Optuna para el valor del regularizador. 
- `opt` permite que se elija un regularizador diferente a los propuestos por defecto. Es necesario que tenga a su vez `custom_optimizer = False` para su funcionamiento.
- `eta_interval` permite la elección del intervalo de búsqueda de Optuna para el valor de la taza de aprendizaje.
- `class_weight` {} para que no se ocupe balanceo de clase. Ocúpese el obtenido con el módulo `data_set_creator` y la función `model_sets(weight_classes = True)` para obtener el diccionario correspondiente. 
- `dropout_interval`permite la elección del intervalo de búsqueda de Optuna para el porcentaje de dropout por capa.
- `custom_optimizer` elíjase como `True`si se desea ocupar un optimizador diferente y que Optuna no explore este hiperparámetro. 

- `objective_tracked()`. Permite realizar experimentos en Optuna y trackear el rendimiento de los experimentos usando la librería de WandB. Las variables son las mismas que en `objective()`.

Es necesario realizar unos pasos extras con Optuna para realizar los experimentos. En [SINBA-example](https://github.com/emmdaz/SINBA-example.git) muestro un ejemplo de cómo hacerlo.

## Rendimiento

En [SINBA-example](https://github.com/emmdaz/SINBA-example.git) se muestra un ejemplo de la implementación de la librería SINBA. En este apartado ha de mostrarse el rendimiento del programa. 

El ejemplo hace uso de dos archivos .ROOT derivados de una simulación Delphes de colisiones de un protón y un antiprotón. El proceso genera un bosón Z y un bosón H (Higgs) que decaen a un par leptónico y un quark b/anti-b y s/anti-s. Al usar el progrma SINBA fue configurado para reconstruir la masa invariante del bosón de Higgs (a partir de sus decaimientos) y la masa invariante entre los pares leptónicos corresponde a los del bosón Z, por ello, la columna `DER_mass_lep` no fue removida. Fue entonces creada una base de datos como la siguiente:

![alt text](image.png)

El programa para crear los conjuntos de entrenamiento, validación y prueba fue configurado para generar estos tres conjuntos sin considerar balanceo de clase y dividiendo los datos en 50% entrenamiento, 30% validación y 20% prueba; teniendo pues un número de datos:

- Train size: 1572
- Validation size: 628
- Test size: 944 

Se realizó un experimento de Optuna en el que exploró 20 configuraciones para los hiperparámetros del modelo con la configuración predeterminada propuesta en `SINBA.objective()`. Luego de estos intentos Optuna encontró que los mejores hiperparámetros para una red residual eran:

```
{'Activation_Function': 'leaky_relu', 'Layer_Regularizer': 'l2', 'Regularizer_Value': 2.9030212081462723e-07, 'N_layers': 17, 'learning_rate': 0.0009406914712140204, 'optimizer': 'adamw', 'N_1st_layer': 233, 'Dropout': 'n', 'Regularizer': 'l2', 'Reg_value': 8.882681102034641e-07, 'N_1_layer': 194, 'N_2_layer': 193, 'N_3_layer': 155, 'N_4_layer': 229, 'N_5_layer': 145, 'N_6_layer': 248, 'N_7_layer': 189, 'N_8_layer': 189, 'N_9_layer': 162, 'N_10_layer': 198, 'N_11_layer': 184, 'N_12_layer': 172, 'N_13_layer': 247, 'N_14_layer': 219, 'N_15_layer': 239, 'N_16_layer': 218, 'N_17_layer': 146}
```

Se diseñó así, la red y fue obtenido al final un rendimiento como sigue:

- loss: 0.0062
- accuracy: 1.0000
- precision: 1.0000
- AUC: 1.0000
- AUPRC: 1.0000

Para los datos de prueba. 

Las gráficas de pérdida durante el entrenamiento y la validación fueron las siguientes:

![alt text](image-1.png)

Las gráficas de exactitud durante el entrenamiento y la validación fueron las siguientes:

![alt text](image-2.png)

Las gráficas de precisión durante el entrenamiento y la validación fueron las siguientes:

![alt text](image-3.png)

Reportándose una matriz de confusión para los datos de validación como sigue:

![alt text](image-4.png)

## Consideraciones importantes

SINBA fue desarrollado para implementarse usando el contenedor de docker `tensorflow-2.15.0-gpu`. Es sugerido enormemente sea implementado de la misma forma (pues así fue cómo fue hecho). 

SINBA requiere otras librerías para su uso. Estas son:
- `Uproot (5.7.1)`
- `Pandas (2.2.2)`
- `Numpy (1.26.2)`
- `Matplotlib (3.8.4)`
- `Seaborn (0.13.2)`
- `Optuna (4.6.0)`
- `WandB (0.22.3)`
- `Opencv-Python-Headless (4.9.0.80)`
- `Keras (2.15.0)`
- `Tensorflow (2.15.0)`
- `Scikit-learn (1.7.2)`

Todo trabajando con `Python 3.11.0rc1`


## Bibliografía

[1]: Balazs Kegl, CecileGermain, ChallengeAdmin, ClaireAdam, David Rousseau, Djabbz, fradav, Glen Cowan, Isabelle, and joycenv. Higgs Boson Machine Learning Challenge. https://kaggle.com/competitions/higgs-boson, 2014. Kaggle.

[^1]: El diseño fue resultado de los experimentos que realizé en mi repositorio [Higgs-Boson-SignalnBackground-Classificator](https://github.com/emmdaz/Higgs-Boson-SignalnBackground-Classificator.git)

[^2]: El programa está configurado actualmente para trabajar únicamente con partículas masivas. La implementación para fotones sigue en curso.
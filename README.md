# _SINBA ResNet_: User Guide

### Abstract 

La clasificación de señal y ruido de eventos de altas energías implica hasta la fecha un reto para la física experimental moderna. Múltiples métodos de aprendizaje de máquina para su diferenciación han sido empleados como lo son los árboles de decisión, los Random Forest o el clustering con un buen desempeño y son el estado del arte más empleado. En este trabajo se propone utilizar un diseño de red neuronal artificial para la clasificación de estos eventos. *_SINBA_* (SIgnal aNd BAckground) es un programa que ayuda a la búsqueda de hiperparámetros para una red neuronal tipo ResNet, que clasifica eventos señal y ruido de eventos de colisiones de partículas simuladas con Delphes. El programa permite elegir la búsqueda de eventos en los que una partícula haya decaído a pares de partículas y reconstruir su masa invariante para crear una base de datos basada en el _Higgs Boson Challenge_ con la que es entrenada la red . Se reporta un ejemplo en el que se entrenó una red para la diferenciación de eventos que produjeron un Higgs que decae a un par leptónico $\ell^+ / \ell^-$ y a un quark b/anti-b $b / \bar{b}$ y un quark anti_s/b $s / \bar{s}$.

## Introducción

El programa SINBA tiene como principal objeto facilitar el diseño de una red neuronal tipo ResNet para realizar la diferenciación de eventos señal y eventos ruido de colisiones de partículas. Su diseño está enfocado a trabajar con archivos .ROOT derivados de simulaciones Delphes. 

El programa requiere la instalación de la paquetería Uproot, la cual es una alternativa para la paquetería PyRoot, y permite trabajar archivos .ROOT con librerías más familiares como Numpy y Pandas. La documentación completa de la paquetería Uproot puede encontrarse en [Uproot Library](https://uproot.readthedocs.io/en/stable/index.html).

SINBA permite el cálculo de variables para el entrenamiento de una red neuronal basándose en las recomendadas en el _Higgs Boson Challenge_ del CERN. Una descripción más detallada sobre cada una de estas variables puede encontrarse en [1]. 

Entre las configuraciones de los programas empleados hay anexada una opción para guardar los DataFrames
creados para su funcionamiento. Esto ha sido diseñado de esta manera para permitir al usuario guardar las bases de datos en el futuro y permitir entonces crear, por ejemplo archivos .csv, que pudieran resultar más cómodos para trabajar con métodos similares o diferentes de aprendizaje de máquina.

## Variables Calculator

El primer módulo esencial para SINBA es variables_calculator. Este programa permite realizar el cálculo de
cantidades importantes a utilizar para el entrenamiento de la red neuronal. Entre estas se encuentan la masa invariante, la masa transversa, la separación de la pseudo-velocidad, el momento transverso, entre otros.

Las variables usadas para el entrenamiento se dividen esencialmente en dos tipos: Las primarias y las derivadas. Como su nombre indica, las primarias son las que se obtienen de la información en crudo del archivo ROOT, e.g. el número de jets en el evento; por su lado, las variables derivadas serán aquellas que se obtienen a partir de cálculos con las primarias. La variables primarias son etiquetadas en la base de datos o DataFrame crado con el prefijo PRI. Similarmente, las variables derivadas son etiquetadas con un prefijo DER.

En las simulaciones Delphes puede permitirse el caso en el que en un evento hay más de una partícula de un tipo, e.g. que haya tres electrones, y en todo caso la elección de la variable será acorde a la correspondiente de la partícula que tenga el mayor valor de momento transverso[^1]. 

Las variables posibles a calcularse usando este módulo son: 

- DER_INV_m: Reconstrucción de la masa invariante de una partícula que decae a otras dos. Se calcula con la función inv_m:

```
def inv_m(file, p1, p2, charge = 0, flavor1 = 0, flavor2 = 0, save_df = False, graph = False, mass = 0)
```
- Variables de la función:
    - File. Archivo .ROOT con el que se trabajará.
    - p1, p2. Partícula 1 y 2 de la que se reconstuirá la masa invariante.
    - charge. Carga teórica de la partícula reconstruida.
    - flavor1. Flavor del primer jet. Configurado por defecto como 0 para que si p1 no es un jet, no se considere entonces su flavor.
    - flavor2. Flavor del segundo jet. Configurado por defecto como 0 para que si p2 no es un jet, no se considere entonces su flavor.
    - save_df. Opción de guardar el DataFrame creado como un archivo .csv.
    - graph = Para graficar la distribución de la masa invariante calculada en los eventos.
    - mass. Para añadir una línea de referencia que indique un punto de la gráfica de la distribución de la masa invariante reconstruida por evento. 

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

- PRI_met y PRI_met_phi. Energía perdida transversa y su ángulo $\phi$. Se calculan con la función met():

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


[^1]: El programa está configurado actualmente para trabajar únicamente con partículas masivas. La implementación para fotones sigue en curso.

[1]: Balazs Kegl, CecileGermain, ChallengeAdmin, ClaireAdam, David Rousseau, Djabbz, fradav, Glen Cowan, Isabelle, and joycenv. Higgs Boson Machine Learning Challenge. https://kaggle.com/competitions/higgs-boson, 2014. Kaggle.
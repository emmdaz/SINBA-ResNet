# _SINBA ResNet_: User Guide

### Abstract 

La clasificación de señal y ruido de eventos de altas energías implica hasta la fecha un reto para la física experimental moderna. Múltiples métodos de aprendizaje de máquina para su diferenciación han sido empleados como lo son los árboles de decisión, los Random Forest o el clustering con un buen desempeño y son el estado del arte más empleado. En este trabajo se propone utilizar un diseño de red neuronal artificial para la clasificación de estos eventos. *_SINBA_* (SIgnal aNd BAckground) es un programa que ayuda a la búsqueda de hiperparámetros para una red neuronal tipo ResNet, que clasifica eventos señal y ruido de eventos de colisiones de partículas simuladas con Delphes. El programa permite elegir la búsqueda de eventos en los que una partícula haya decaído a pares de partículas y reconstruir su masa invariante para crear una base de datos basada en el _Higgs Boson Challenge_ con la que es entrenada la red . Se reporta un ejemplo en el que se entrenó una red para la diferenciación de eventos que produjeron un Higgs que decae a un par leptónico $\ell^+ / \ell^-$ y a un quark b/anti-b $b / \bar{b}$ y un quark anti_s/b $s / \bar{s}$.

## Introducción

El programa SINBA tiene como principal objeto facilitar el diseño de una red neuronal tipo ResNet para realizar la diferenciación de eventos señal y eventos ruido de colisiones de partículas. Su diseño está enfocado a trabajar con archivos .ROOT derivados de simulaciones Delphes. 

El programa requiere la instalación de la paquetería Uproot, la cual es una alternativa para la paquetería PyRoot, y permite trabajar archivos .ROOT con librerías más familiares como Numpy y Pandas. La documentación completa de la paquetería Uproot puede encontrarse en [Uproot Library](https://uproot.readthedocs.io/en/stable/index.html).

SINBA permite el cálculo de variables para el entrenamiento de una red neuronal basándose en las recomendadas en el _Higgs Boson Challenge_ del CERN. Una descripción más detallada sobre cada una de estas variables puede encontrarse en [1]. 

Entre las configuraciones de los programas empleados hay anexada una opción para guardar los DataFrames
creados para su funcionamiento. Esto ha sido diseñado de esta manera para permitir al usuario guardar las bases de datos en el futuro y permitir entonces crear, por ejemplo archivos .csv, que pudieran resultar más cómodos para trabajar con métodos similares o diferentes de aprendizaje de máquina.

[1] Balazs Kegl, CecileGermain, ChallengeAdmin, ClaireAdam, David Rousseau, Djabbz, fradav, Glen Cowan, Isabelle, and joycenv. Higgs Boson Machine Learning Challenge. https://kaggle.com/competitions/higgs-boson, 2014. Kaggle.
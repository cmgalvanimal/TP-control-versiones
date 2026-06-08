import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from carga_datos import CargaDatos

datos = CargaDatos("data/jugadores_2024_2025.csv").cargar_datos()

"""
Modelo 1:

Variable respuesta:
- Gls: cantidad de goles convertidos por cada jugador.

Variables predictoras:
- SoT: cantidad de tiros al arco.
- xG: cantidad de goles esperados.
- Min: cantidad de minutos jugados.
"""

y = datos['Gls'] # respuesta    

variables_multiple_1 = ["SoT", "xG", "Min"]
X_multiple_1 = sm.add_constant(datos[variables_multiple_1])

modelo_multiple_1 = sm.OLS(y, X_multiple_1).fit()
print(modelo_multiple_1.summary())

#R^2 ajustado del modelo 1
r2_M1= modelo_multiple_1.rsquared_adj
print(r2_M1)


"""
Modelo 2:

Variable respuesta:
- Gls: cantidad de goles convertidos por cada jugador.

Variables predictoras:
- SoT: cantidad de tiros al arco.
- xG: cantidad de goles esperados.
"""

variables_multiple_2 = ["SoT", "xG"]
X_multiple_2 = sm.add_constant(datos[variables_multiple_2])

modelo_multiple_2 = sm.OLS(y, X_multiple_2).fit()
print(modelo_multiple_2.summary())

#R^2 ajustado del modelo 2
r2_M2= modelo_multiple_2.rsquared_adj
print(r2_M2)


"""
Modelo 3:

Variable respuesta:
- Gls: cantidad de goles convertidos por cada jugador.

Variables predictoras:
- Min: cantidad de minutos jugados.
- xG: cantidad de goles esperados.
"""

variables_multiple_3 = ["xG", "Min"]
X_multiple_3 = sm.add_constant(datos[variables_multiple_3])

modelo_multiple_3 = sm.OLS(y, X_multiple_3).fit()
print(modelo_multiple_3.summary())

#R^2 ajustado del modelo 3
r2_M3= modelo_multiple_3.rsquared_adj   
print(r2_M3)


"""
Modelo 4:

Variable respuesta:
- Gls: cantidad de goles convertidos por cada jugador.

Variables predictoras:
- SoT: cantidad de tiros al arco.
- Sh/90: cantidad de tiros realizados por cada 90 minutos jugados.
"""

variables_multiple_4 = ["SoT", "Sh/90"]
X_multiple_4 = sm.add_constant(datos[variables_multiple_4])

modelo_multiple_4 = sm.OLS(y, X_multiple_4).fit()
print(modelo_multiple_4.summary())

#R^2 ajustado del modelo 4
r2_M4= modelo_multiple_4.rsquared_adj   
print(r2_M4)



"""
Verificación de supuestos: Normalidad 
"""


#modelo 1 --> test normalidad
residuos = modelo_multiple_1.resid

x_ord = np.sort(residuos)
media = np.mean(x_ord)
desvio = np.std(x_ord, ddof=1)
x_ord_s = (x_ord - media) / desvio

n = len(x_ord)
p = np.arange(1, n + 1) / (n + 1)
cuantiles_teoricos = norm.ppf(p)

fig, ax = plt.subplots(2,2, figsize=(14, 12))

ax[0, 0].scatter(cuantiles_teoricos, x_ord_s)
ax[0, 0].plot(cuantiles_teoricos, cuantiles_teoricos, color="red", linestyle="--")
ax[0, 0].set_xlabel("Cuantiles teóricos")
ax[0, 0].set_ylabel("Residuos estandarizados")
ax[0, 0].set_title("QQ-plot Modelo 1")
ax[0, 0].grid(True)

#modelo 2 --> test normalidad
residuos = modelo_multiple_2.resid

x_ord = np.sort(residuos)
media = np.mean(x_ord)
desvio = np.std(x_ord, ddof=1)
x_ord_s = (x_ord - media) / desvio

n = len(x_ord)
p = np.arange(1, n + 1) / (n + 1)
cuantiles_teoricos = norm.ppf(p)

ax[0, 1].scatter(cuantiles_teoricos, x_ord_s)
ax[0, 1].plot(cuantiles_teoricos, cuantiles_teoricos, color="red", linestyle="--")
ax[0, 1].set_xlabel("Cuantiles teóricos")
ax[0, 1].set_ylabel("Residuos estandarizados")
ax[0, 1].set_title("QQ-plot Modelo 2")
ax[0, 1].grid(True)

#modelo 3 --> test normalidad
residuos = modelo_multiple_3.resid

x_ord = np.sort(residuos)
media = np.mean(x_ord)
desvio = np.std(x_ord, ddof=1)
x_ord_s = (x_ord - media) / desvio

n = len(x_ord)
p = np.arange(1, n + 1) / (n + 1)
cuantiles_teoricos = norm.ppf(p)

ax[1, 0].scatter(cuantiles_teoricos, x_ord_s)
ax[1, 0].plot(cuantiles_teoricos, cuantiles_teoricos, color="red", linestyle="--")
ax[1, 0].set_xlabel("Cuantiles teóricos")
ax[1, 0].set_ylabel("Residuos estandarizados")
ax[1, 0].set_title("QQ-plot Modelo 3")
ax[1, 0].grid(True)

#modelo 4 --> test normalidad
residuos = modelo_multiple_4.resid

x_ord = np.sort(residuos)
media = np.mean(x_ord)
desvio = np.std(x_ord, ddof=1)
x_ord_s = (x_ord - media) / desvio

n = len(x_ord)
p = np.arange(1, n + 1) / (n + 1)
cuantiles_teoricos = norm.ppf(p)

ax[1, 1].scatter(cuantiles_teoricos, x_ord_s)
ax[1, 1].plot(cuantiles_teoricos, cuantiles_teoricos, color="red", linestyle="--")
ax[1, 1].set_xlabel("Cuantiles teóricos")
ax[1, 1].set_ylabel("Residuos estandarizados")
ax[1, 1].set_title("QQ-plot Modelo 4")
ax[1, 1].grid(True)

plt.show()


"""
Verificación de supuestos: Homocedasticidad
"""

#homocedasticidad modelo 1
y_hat = modelo_multiple_1.predict(X_multiple_1)
residuos = modelo_multiple_1.resid

fig, ax = plt.subplots(2,2, figsize=(14,12))

ax[0, 0].scatter(y_hat, residuos)
ax[0, 0].axhline(0, color="red", linestyle="--")
ax[0, 0].set_xlabel("Valores predichos (ŷ)")
ax[0, 0].set_ylabel("Residuos (y - ŷ)")
ax[0, 0].set_title("Residuos vs valores predichos Modelo 1")
ax[0, 0].grid(True)

#homocedasticidad modelo 2
y_hat = modelo_multiple_2.predict(X_multiple_2)
residuos = modelo_multiple_2.resid

ax[0, 1].scatter(y_hat, residuos)
ax[0, 1].axhline(0, color="red", linestyle="--")
ax[0, 1].set_xlabel("Valores predichos (ŷ)")
ax[0, 1].set_ylabel("Residuos (y - ŷ)")
ax[0, 1].set_title("Residuos vs valores predichos Modelo 2")
ax[0, 1].grid(True)

#homocedasticidad modelo 3
y_hat = modelo_multiple_3.predict(X_multiple_3)
residuos = modelo_multiple_3.resid

ax[1, 0].scatter(y_hat, residuos)
ax[1, 0].axhline(0, color="red", linestyle="--")
ax[1, 0].set_xlabel("Valores predichos (ŷ)")
ax[1, 0].set_ylabel("Residuos (y - ŷ)")
ax[1, 0].set_title("Residuos vs valores predichos Modelo 3")
ax[1, 0].grid(True)

#homocedasticidad modelo 4
y_hat = modelo_multiple_4.predict(X_multiple_4)
residuos = modelo_multiple_4.resid

ax[1, 1].scatter(y_hat, residuos)
ax[1, 1].axhline(0, color="red", linestyle="--")
ax[1, 1].set_xlabel("Valores predichos (ŷ)")
ax[1, 1].set_ylabel("Residuos (y - ŷ)")
ax[1, 1].set_title("Residuos vs valores predichos Modelo 4")
ax[1, 1].grid(True)


plt.show()

"""
Comparación de modelos con R^2 ajustado:

Los modelos M1 y M2 tienen valores de R^2 ajustado prácticamente iguales (0.888 y 0.887), por lo que ambos explican casi la misma proporción de la variabilidad de la cantidad de goles. Sin embargo, M2 utiliza solamente las variables SoT y xG, mientras que M1 incorpora además la variable Min. Como al incorporar  Min se produce una mejora muy pequeña en el ajuste del modelo, resulta más conveniente elegir M2. Al tener menos variables, es un modelo más simple y fácil de interpretar, sin perder capacidad explicativa. Por lo tanto, M2 representa una mejor elección entre ambos modelos.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from carga_datos import CargaDatos

datos = CargaDatos("data/jugadores_2024_2025.csv").cargar_datos()
datos_filtrados = datos[(datos['Gls'] >= 1) & (datos['Gls'] <= 10)].copy()
y_filtrado = datos_filtrados['Gls'] # respuesta
variables_m2 = ["SoT", "xG"]
X_m2_filtrado = sm.add_constant(datos_filtrados[variables_m2])
modelo_final = sm.OLS(y_filtrado, X_m2_filtrado).fit()
print(modelo_final.summary())

residuos = modelo_final.resid

x_ord = np.sort(residuos)
media = np.mean(x_ord)
desvio = np.std(x_ord, ddof=1)
x_ord_s = (x_ord - media) / desvio

n = len(x_ord)
p = np.arange(1, n + 1) / (n + 1)
cuantiles_teoricos = norm.ppf(p)

fig, ax = plt.subplots(1, 2, figsize=(14,6))
ax[0].scatter(cuantiles_teoricos, x_ord_s)
ax[0].plot(cuantiles_teoricos, cuantiles_teoricos, color="red", linestyle="--")
ax[0].set_xlabel("Cuantiles teóricos")
ax[0].set_ylabel("Residuos estandarizados")
ax[0].set_title("QQ-plot Modelo 2 - Sin extremos")
ax[0].grid(True)

#homocedasticidad modelo 2
y_hat = modelo_final.predict(X_m2_filtrado)
residuos = modelo_final.resid

ax[1].scatter(y_hat, residuos)
ax[1].axhline(0, color="red", linestyle="--")
ax[1].set_xlabel("Valores predichos (ŷ)")
ax[1].set_ylabel("Residuos (y - ŷ)")
ax[1].set_title("Residuos vs valores predichos")
ax[1].grid(True)

plt.show()

"El $72.7\%$ de la variabilidad de los goles de este grupo sigue estando fuertemente explicada por solo dos variables: la cantidad de tiros al arco (SoT) y la calidad de los mismos (xG). Ambos coeficientes resultaron altamente significativos con p-valores de $0.000$."

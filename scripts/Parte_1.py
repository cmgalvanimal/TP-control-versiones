import sys
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

ruta = r"data\jugadores_2024_2025.csv"
df = pd.read_csv(ruta, delimiter=",")

# Variable observada
goles = df["Gls"]

# Posibles variables predictoras según criterios lógicos
nacionalidad = df["Nation"]
posicion = df["Pos"]
edad = df["Age"]
minutos = df["Min"]
tiros_tot = df["Sh"]
tiros_arco = df["SoT"]
goles_esp = df["xG"]
pases = df["Cmp"]
intercepciones = df["Int"]
acciones = df["SCA"]
conduc = df["Carries"]
conduc_prog = df["PrgC"]

# Análisis visual
fig, ax = plt.subplots(4, 3)
ax[0,0].scatter(nacionalidad.astype("category").cat.codes, goles, s=10)
ax[0,0].set_title("vs nacionalidad")
ax[0,1].scatter(posicion.astype("category").cat.codes, goles, s=10)
ax[0,1].set_title("vs posicion")
ax[0,2].scatter(edad, goles, s=10)
ax[0,2].set_title("vs edad")
ax[1,0].scatter(minutos, goles, s=10)
ax[1,0].set_title("vs minutos")
ax[1,1].scatter(tiros_tot, goles, s=10)
ax[1,1].set_title("vs tiros_tot")
ax[1,2].scatter(tiros_arco, goles, s=10)
ax[1,2].set_title("vs tiros_arco")
ax[2,0].scatter(goles_esp, goles, s=10)
ax[2,0].set_title("vs goles_esp")
ax[2,1].scatter(pases, goles, s=10)
ax[2,1].set_title("vs pases")
ax[2,2].scatter(intercepciones, goles, s=10)
ax[2,2].set_title("vs intercepciones")
ax[3,0].scatter(acciones, goles, s=10)
ax[3,0].set_title("vs acciones")
ax[3,1].scatter(conduc, goles, s=10)
ax[3,1].set_title("vs conduc")
ax[3,2].scatter(conduc_prog, goles, s=10)
ax[3,2].set_title("vs conduc_prog")
plt.tight_layout()
plt.show()

# Hay 3 variables que parecen presentar un comportamiento lineal respecto a
# "goles": "tiros_tot", "tiros_arco", "goles_esp".

# Eliminar filas incompletas (en las 4 columnas mencionadas)
n = len(goles)
for i in range(n):
    if (
        np.isnan(goles.iloc[i]) or np.isnan(tiros_tot.iloc[i]) or
        np.isnan(tiros_arco.iloc[i]) or np.isnan(goles_esp.iloc[i])
    ):
        goles.drop(goles.index[i], inplace=True)
        tiros_tot.drop(tiros_tot.index[i], inplace=True)
        tiros_arco.drop(tiros_arco.index[i], inplace=True)
        goles_esp.drop(goles_esp.index[i], inplace=True)

# Coeficientes de correlación para las 3 variables mencionadas
print("coeficiente de correlación tiros_tot: ", tiros_tot.corr(goles))
print("coeficiente de correlación tiros_arco: ", tiros_arco.corr(goles))
print("coeficiente de correlación goles_esp: ", goles_esp.corr(goles))

# La mejor correlación la presenta "goles_esp". Se escoge esta variable

# Ajuste lineal
X = sm.add_constant(goles_esp)
modelo = sm.OLS(goles, X)
resultado = modelo.fit()
a = resultado.params.iloc[1]
b = resultado.params.iloc[0]

# p-valor bajo la hipótesis de que la pendiente es nula
print("p-valor para la pendiente: ", resultado.pvalues.iloc[1])

# Gráfico de puntos con recta de ajuste
fig, ax = plt.subplots()
ax.scatter(goles_esp, goles, color="blue")
ax.axline((0,b), (1,a+b), color="red")
ax.set_title("Goles marcados vs goles esperados")
ax.set_xlabel("Goles esperados")
ax.set_ylabel("Goles marcados")
plt.show()
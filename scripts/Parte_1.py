import sys
import pandas as pd
import matplotlib.pyplot as plt

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
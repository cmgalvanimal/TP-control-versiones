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
minutos = df["Min"]
tiros_tot = df["Sh"]
tiros_arco = df["SoT"]
goles_esp = df["xG"]
pases = df["Cmp"]
intercepc = df["Int"]
acciones = df["SCA"]
conduc = df["Carries"]
conduc_p = df["PrgC"]

# Eliminar filas incompletas
n = len(goles)
for i in range(n):
    if (
        np.isnan(goles.iloc[i]) or np.isnan(minutos.iloc[i]) or
        np.isnan(tiros_tot.iloc[i]) or np.isnan(tiros_arco.iloc[i]) or
        np.isnan(goles_esp.iloc[i]) or np.isnan(pases.iloc[i]) or
        np.isnan(intercepc.iloc[i]) or np.isnan(acciones.iloc[i]) or
        np.isnan(conduc.iloc[i]) or np.isnan(conduc_p.iloc[i])
    ):
        goles.drop(goles.index[i], inplace=True)
        minutos.drop(minutos.index[i], inplace=True)
        tiros_tot.drop(tiros_tot.index[i], inplace=True)
        tiros_arco.drop(tiros_arco.index[i], inplace=True)
        goles_esp.drop(goles_esp.index[i], inplace=True)
        pases.drop(pases.index[i], inplace=True)
        intercepc.drop(intercepc.index[i], inplace=True)
        acciones.drop(acciones.index[i], inplace=True)
        conduc.drop(conduc.index[i], inplace=True)
        conduc_p.drop(conduc_p.index[i], inplace=True)
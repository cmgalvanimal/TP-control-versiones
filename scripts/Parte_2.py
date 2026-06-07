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
        minutos.drop(minutos.index[i], inplace=True)#
        tiros_tot.drop(tiros_tot.index[i], inplace=True)
        tiros_arco.drop(tiros_arco.index[i], inplace=True)
        goles_esp.drop(goles_esp.index[i], inplace=True)
        pases.drop(pases.index[i], inplace=True)#
        intercepc.drop(intercepc.index[i], inplace=True)#
        acciones.drop(acciones.index[i], inplace=True)
        conduc.drop(conduc.index[i], inplace=True)#
        conduc_p.drop(conduc_p.index[i], inplace=True)#

# Ajuste multilineal 1, 9 variables
x = np.stack((
    minutos, tiros_tot, tiros_arco, goles_esp, pases,
    intercepc, acciones, conduc, conduc_p
), axis=1)
X = sm.add_constant(x)
mod1 = sm.OLS(goles, X)
res1 = mod1.fit()
print("p-valor minutos: ", res1.pvalues.iloc[1])
print("p-valor tiros_tot: ", res1.pvalues.iloc[2])
print("p-valor tiros_arco: ", res1.pvalues.iloc[3])
print("p-valor goles_esp: ", res1.pvalues.iloc[4])
print("p-valor pases: ", res1.pvalues.iloc[5])
print("p-valor intercepc: ", res1.pvalues.iloc[6])
print("p-valor acciones: ", res1.pvalues.iloc[7])
print("p-valor conduc: ", res1.pvalues.iloc[8])
print("p-valor conduc_p: ", res1.pvalues.iloc[9])

# Dados los p-valores obtenidos, se seleccionan las siguientes variables:
# tiros_tot, tiros_arco, goles_esp, acciones

# Ajuste multilineal 2, 4 variables
x = np.stack((tiros_tot, tiros_arco, goles_esp, acciones), axis=1)
X = sm.add_constant(x)
mod2 = sm.OLS(goles, X)
res2 = mod2.fit()
print("----------------")
print("p-valor tiros_tot: ", res2.pvalues.iloc[1])
print("p-valor tiros_arco: ", res2.pvalues.iloc[2])
print("p-valor goles_esp: ", res2.pvalues.iloc[3])
print("p-valor acciones: ", res2.pvalues.iloc[4])

# Dados los p-valores obtenidos se rechaza la hipótesis de que
# alguna de las 4 pendientes sea nula
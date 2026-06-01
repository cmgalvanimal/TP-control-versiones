import statsmodels.api as sm
import pandas as pd
import numpy as np
from pathlib import Path
import sys


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

from src.carga_datos import CargaDatos
ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)
df = cargador.cargar_datos()

# Regresion Lineal Simple.
'''Para predecir la cantidad de goles, tomaremos las variables Sh (tiros totales), SoT (tiros al arco) 
y Min (minutos jugados) para hacer, por separado, cada una de las regresiones lineales simples.'''

y= df['Gls']
# Variable predictora: tiros totales
x = np.array(df['Sh'])
X = sm.add_constant(x)
modelo1 = sm.OLS(y, X)
result1 = modelo1.fit()
print(result1)
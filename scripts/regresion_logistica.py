from pathlib import Path
import sys
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

from src.carga_datos import CargaDatos

ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

df = cargador.cargar_datos()

y = 1*(df['Gls'] > 0)
"""Creamos un vector booleano que revisa si el jugador metió un gol o no"""
p_sombrero = np.mean(y) #Proporción de todos los jugadores que metieron un gol


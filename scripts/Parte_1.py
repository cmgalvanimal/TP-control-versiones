from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))
from src.carga_datos import CargaDatos
from src.RegresionLineal_v2 import RegresionLinealSimple

ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)
cargador.cargar_datos()
data = cargador.obtener_datos()

x = data['xG']
y = data['G+A']

modelo = RegresionLinealSimple(x, y)
modelo.graficar_recta_ajustada()
modelo.graficar_qqplot()
plt.show()
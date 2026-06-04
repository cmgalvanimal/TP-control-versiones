import statsmodels.api as sm
from pathlib import Path
import sys
from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

from src.carga_datos import CargaDatos
ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

cargador.cargar_datos()
cargador.mostrar_dimensiones()

jugadores=cargador.datos
print(jugadores.head())


class regresion_lineal:
    def __init__(self, datos):
        # Ahora se recibe directamente el DataFrame y se guarda en self.datos
        self.datos = datos

        # Inicializamos las variables del modelo
        self.y_1 = None
        self.x_1 = None 
        self.modelo = None  

    def regresion_lineal_simple(self, y, x):
        self.y_1 = self.datos[y] 
        self.x_1 = self.datos[x]
        self.X_1 = sm.add_constant(self.x_1)
        self.modelo = sm.OLS(self.y_1, self.X_1).fit()
        return self.modelo.summary()

    def qqplot_residuos(self):
        # Controlamos que el modelo ya exista antes de graficar
        if self.modelo is None:
            raise ValueError("Primero debe ajustar el modelo con regresion_lineal_simple().")

        residuos = self.modelo.resid
        residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)
        res_ordenados = np.sort(residuos_std)

        n = len(res_ordenados)
        proporciones = np.linspace(0.5 / n, 1 - 0.5 / n, n)
        cuantiles_teoricos = stats.norm.ppf(proporciones)

        plt.figure(figsize=(8, 6))
        plt.scatter(cuantiles_teoricos, res_ordenados)
        mn = min(cuantiles_teoricos)
        mx = max(cuantiles_teoricos)
        plt.plot([mn, mx], [mn, mx], color="red")
        plt.title("Q-Q Plot de Residuos")
        plt.xlabel("Cuantiles teóricos")
        plt.ylabel("Residuos estandarizados (ordenados)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def grafico_ajuste(self):
        # Controlamos que el modelo ya exista antes de graficar
        if self.modelo is None:
            raise ValueError("Primero debe ajustar el modelo con regresion_lineal_simple().")

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.x_1,
            self.y_1,
            color="steelblue",
            s=40,
            alpha=0.7,
            edgecolor="white",
            label="Datos observados",
        )

        x_ordenado = np.sort(self.x_1)
        X_ordenado_const = sm.add_constant(x_ordenado)
        y_pred = self.modelo.predict(X_ordenado_const)

        plt.plot(
            x_ordenado,
            y_pred,
            color="firebrick",
            linewidth=2.5,
            label="Recta ajustada (Modelo)",
        )

        plt.title("Ajuste de Regresión Lineal Simple", fontsize=12)
        plt.xlabel(f"Variable Predictora ({self.x_1.name})")
        plt.ylabel(f"Variable Respuesta ({self.y_1.name})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()




modelo_goles_esperados = regresion_lineal(jugadores)
print(modelo_goles_esperados.regresion_lineal_simple("Gls", "SoT"))
modelo_goles_esperados.grafico_ajuste()
modelo_goles_esperados.qqplot_residuos()

modelo_tiros_al_arco = regresion_lineal(jugadores)
print(modelo_tiros_al_arco.regresion_lineal_simple("Gls", "xG"))
modelo_tiros_al_arco.grafico_ajuste()
modelo_tiros_al_arco.qqplot_residuos()
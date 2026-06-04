import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


class ModeloRegresionSimple:
    def __init__(self, df, variable_x, variable_y):
        self.df = df

        datos = df[[variable_x, variable_y]].dropna()

        self.X = sm.add_constant(datos[variable_x])
        self.y = datos[variable_y]

        self.modelo = sm.OLS(self.y, self.X).fit()

        self.variable_x = variable_x
        self.variable_y = variable_y

    def summary(self):
        return self.modelo.summary()

    @property
    def r2(self):
        return self.modelo.rsquared

    @property
    def pvalue(self):
        return self.modelo.pvalues[self.variable_x]

    def grafico_regresion(self):
        x = self.X[self.variable_x]
        y = self.y

        plt.figure(figsize=(10, 6))

        plt.scatter(
            x,
            y,
            alpha=0.5,
        )

        x_line = np.linspace(x.min(), x.max(), 100)

        y_line = (
            self.modelo.params["const"] + self.modelo.params[self.variable_x] * x_line
        )

        plt.plot(
            x_line,
            y_line,
            linewidth=3,
            label="Recta de regresión",
        )

        plt.xlabel(self.variable_x)
        plt.ylabel(self.variable_y)

        plt.title(f"Relación entre {self.variable_x} y {self.variable_y}")

        plt.legend()
        plt.grid(True)

        plt.show()

    def grafico_residuos(self):
        residuos = self.modelo.resid
        ajustados = self.modelo.fittedvalues

        plt.figure(figsize=(10, 6))

        plt.scatter(
            ajustados,
            residuos,
            alpha=0.5,
        )

        plt.axhline(
            y=0,
            linestyle="--",
        )

        plt.xlabel("Valores ajustados")
        plt.ylabel("Residuos")

        plt.title("Residuos vs Valores ajustados")

        plt.grid(True)

        plt.show()

    def histograma_residuos(self):
        residuos = self.modelo.resid

        plt.figure(figsize=(10, 6))

        plt.hist(
            residuos,
            bins=30,
        )

        plt.xlabel("Residuo")
        plt.ylabel("Frecuencia")

        plt.title("Distribución de residuos")

        plt.grid(True)

        plt.show()

    def graficar(self):
        self.grafico_regresion()
        # self.grafico_residuos()
        # self.histograma_residuos()


class BusquedaModelo:
    def __init__(self, df, objetivo="Gls"):
        self.df = df
        self.objetivo = objetivo

    def buscar_mejor_modelo(self):
        resultados = []

        columnas_numericas = self.df.select_dtypes(include=["number"]).columns

        for columna in columnas_numericas:
            if columna == self.objetivo:
                continue

            try:
                datos = self.df[[columna, self.objetivo]].dropna()

                X = sm.add_constant(datos[columna])
                y = datos[self.objetivo]

                modelo = sm.OLS(y, X).fit()

                resultados.append(
                    {
                        "Variable": columna,
                        "R2": modelo.rsquared,
                        "pvalor": modelo.pvalues[columna],
                    }
                )

            except Exception as e:
                print(f"Error en {columna}: {e}")

        resultados = sorted(
            resultados,
            key=lambda x: x["R2"],
            reverse=True,
        )

        print("\n=== Ranking de modelos ===\n")

        for r in resultados:
            print(f"{r['Variable']:15} R²={r['R2']:.4f} p={r['pvalor']:.4e}")

        mejor_variable = resultados[0]["Variable"]

        print(f"\nMejor variable encontrada: {mejor_variable}")

        return ModeloRegresionSimple(
            self.df,
            mejor_variable,
            self.objetivo,
        )

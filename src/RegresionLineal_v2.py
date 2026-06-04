import statsmodels.api as sm
import matplotlib.pyplot as plt 
import numpy as np

class RegresionLineal:
    def __init__(self, x, y):
        if len(x) != len(y):
          raise ValueError('x & y deben tener la misma lonhitud de datos')
        self.x = x
        self.y = y
        self.betas = None
        self.resultado = None

    def ajustar_modelo(self):
        X = sm.add_constant(self.x)
        self.resultado = sm.OLS(self.y, X).fit()
        self.betas = self.resultado.params
        return self.betas

    def graficar_qqplot(self, ax=None):
        import scipy.stats as stats
        if self.resultado is None:  
            self.ajustar_modelo()

        if ax is None:
            fig, ax = plt.subplots()

        datos = np.sort(self.resultado.resid)
        n = len(self.x)
        p = np.arange(1, n + 1) / (n + 1)

        media_est  = np.mean(datos)
        desvio_est = np.std(datos, ddof=1)
        q_teoricos  = stats.norm.ppf(p, loc=media_est, scale=desvio_est)
        q_muestrales = datos

        ax.scatter(q_teoricos, q_muestrales, label="Datos", s=5)

        minimo = min(np.percentile(q_teoricos, 1),  np.percentile(q_muestrales, 1))
        maximo = max(np.percentile(q_teoricos, 99), np.percentile(q_muestrales, 99))

        ax.plot([minimo, maximo], [minimo, maximo], "r--", label="identidad")
        ax.set_xlabel("Cuantiles teoricos")
        ax.set_ylabel("Cuantiles muestrales")
        ax.set_xlim(minimo, maximo)
        ax.set_ylim(minimo, maximo)
        ax.legend()
        return ax

    def obtener_estadisticas(self):
        if self.resultado is None:
            self.ajustar_modelo()

        ic = self.resultado.conf_int()
        return {
            'p_valor':        list(self.resultado.pvalues),
            'error_estandar': list(self.resultado.bse),
            'IC_inf':         list(ic[0]),
            'IC_sup':         list(ic[1]),
        }

    def predecir(self, new_x):
        if self.resultado is None:
            self.ajustar_modelo()
        new_X = sm.add_constant(new_x, has_constant='add')
        return self.resultado.predict(new_X)

    def calcular_intervalos(self, x_new, nivel):
        if self.resultado is None:
            self.ajustar_modelo()
        alpha = 1 - nivel
        x_new = np.atleast_2d(x_new) if np.ndim(x_new) > 1 else np.atleast_1d(x_new)  #prevenir errores con escalares
        X_new = sm.add_constant(x_new, has_constant='add')
        pred_obj = self.resultado.get_prediction(X_new)
        ic_media = pred_obj.conf_int(alpha=alpha, obs=False)
        ip       = pred_obj.conf_int(alpha=alpha, obs=True)
        return {
            "IC_media_inf": list(ic_media[:, 0]),
            "IC_media_sup": list(ic_media[:, 1]),
            "IP_inf":       list(ip[:, 0]),
            "IP_sup":       list(ip[:, 1]),
        }

    def obtener_R2(self):
        if self.resultado is None:
            self.ajustar_modelo()
        return self.resultado.rsquared

    def plot_residuos_vs_predichos(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        if self.resultado is None:
            self.ajustar_modelo()
        
        y_hat = self.resultado.fittedvalues
        residuos = self.resultado.resid
        
        ax.scatter(y_hat,residuos, label="Residuos VS Valores Predichos", facecolor="None", color="b")
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Valores Predichos")
        ax.set_ylabel("Residuos")

        return ax

# ── SIMPLE ────────────────────────────────────────────────────────────────────
class RegresionLinealSimple(RegresionLineal):
    def __init__(self, x, y):
        super().__init__(x, y)

    def predecir(self, new_x):
        """
        new_x puede ser un escalar o un array 1-D.
        Reutiliza el predecir() de la clase base, que ya maneja
        el agregado de la constante con has_constant='add'.
        """
        if self.resultado is None:
            self.ajustar_modelo()

        new_x = np.atleast_1d(new_x)          # garantiza array 1-D
        new_X = sm.add_constant(new_x, has_constant='add')
        return self.resultado.predict(new_X)

    def graficar_recta_ajustada(self, ax=None):
        """
        Grafica los puntos originales y la recta de regresión ajustada.
        Devuelve el Axes para poder componer con otros gráficos (ej: Q Q-plot).
        """
        if self.resultado is None:
            self.ajustar_modelo()

        if ax is None:
            fig, ax = plt.subplots()

        # Puntos observados
        ax.scatter(self.x, self.y, label="Datos observados", s=20, alpha=0.7)

        # Recta ajustada: predicción sobre el rango de x
        x_linea = np.linspace(np.min(self.x), np.max(self.x), 200)
        y_linea  = self.predecir(x_linea)

        beta0, beta1 = self.betas.iloc[0], self.betas.iloc[1]
        ax.plot(
            x_linea, y_linea, "r-",
            label=f"Recta ajustada: ŷ = {beta0:.3f} + {beta1:.3f}·x"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Regresión Lineal Simple — Recta Ajustada")
        ax.legend()
        return ax


# ── MÚLTIPLE ──────────────────────────────────────────────────────────────────
class RegresionLinealMultiple(RegresionLineal):
    def __init__(self, x, y):
        """
        x debe ser una matriz (n × p): n observaciones, p predictores.
        Puede pasarse como lista de listas, ndarray o DataFrame de pandas.
        """
        super().__init__(x, y)

    def predecir(self, new_x):
        """
        new_x: array-like de forma (m, p) — m nuevas observaciones, p predictores.
        Agrega la constante y delega en resultado.predict() de statsmodels.
        """
        if self.resultado is None:
            self.ajustar_modelo()

        new_x = np.atleast_2d(new_x)          # garantiza matriz 2-D
        new_X = sm.add_constant(new_x, has_constant='add')
        return self.resultado.predict(new_X)
    
    def obtener_R2_ajustado(self):
        if self.resultado is None:
            self.ajustar_modelo()
        return self.resultado.rsquared_adj
    
    
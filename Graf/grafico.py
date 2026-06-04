# grafico del Modelo 2
import matplotlib.pyplot as plt

# predichos vs reales
predichos = modelo_multiple_2.predict(X_multiple_2)
reales = datos["Gls"]

plt.scatter(predichos, y)
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()]
)
plt.xlabel("Predichos")
plt.ylabel("Reales")
plt.title("Modelo Múltiple")
plt.show()
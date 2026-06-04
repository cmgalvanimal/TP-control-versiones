import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

datos = pd.read_csv("data\jugadores_2024_2025.csv")
datos["anoto"] = (datos["Gls"] >= 1).astype(int)

variables_logisticas = ["SoT", "xG"]
X_log = sm.add_constant(datos[variables_logisticas])
y_log = datos["anoto"]

n = len(datos)
n_train = int(n*0.8)
n_test = n - n_train
random.seed(10)
indices = random.sample(range(n), n_train)
datos_test = datos.drop(indices)
datos_train = datos.iloc[indices]

#print(datos_train["anoto"].value_counts(normalize=True))
#print(datos_test["anoto"].value_counts(normalize=True))

variables_logisticas = ["SoT", "xG"]
X_train = sm.add_constant(datos_train[variables_logisticas])
y_train = datos_train["anoto"]

X_test = sm.add_constant(datos_test[variables_logisticas])
y_test = datos_test["anoto"]

modelo_log = sm.Logit(y_train, X_train).fit()
print(modelo_log.summary())
probabilidades = modelo_log.predict(X_test)

puntos_corte = np.linspace(0, 1, 101)
sensibilidades = []
especificidades = []

for p in puntos_corte:
    y_pred_temp = (probabilidades >= p).astype(int)
    VP = np.sum((y_pred_temp == 1) & (y_test == 1))
    FP = np.sum((y_pred_temp == 1) & (y_test == 0))
    FN = np.sum((y_pred_temp == 0) & (y_test == 1))
    VN = np.sum((y_pred_temp == 0) & (y_test == 0))
    sensibilidad = VP / (VP + FN) if (VP + FN) > 0 else 0
    especificidad = VN / (VN + FP) if (VN + FP) > 0 else 0
    sensibilidades.append(sensibilidad)
    especificidades.append(especificidad)

sensibilidades = np.array(sensibilidades)
especificidades = np.array(especificidades)
youden = sensibilidades + especificidades - 1
idx_optimo = np.argmax(youden)
p_optimo = puntos_corte[idx_optimo]

y_pred = (probabilidades >= p_optimo).astype(int)

matriz = confusion_matrix(y_test, y_pred, labels=[1, 0])

matriz_confusion = pd.DataFrame(
    matriz,
    index=["Real anoto", "Real no anoto"],
    columns=["Predicho anoto", "Predicho no anoto"]
)

print("Punto de corte optimo:", p_optimo)
print(matriz_confusion)

VP = matriz[0, 0]
FN = matriz[0, 1]
FP = matriz[1, 0]
VN = matriz[1, 1]

sensibilidad = VP / (VP + FN) if (VP + FN) > 0 else 0
especificidad = VN / (VN + FP) if (VN + FP) > 0 else 0
error = (FP + FN) / len(y_test)
auc = roc_auc_score(y_test, probabilidades)

print("Sensibilidad:", round(sensibilidad, 4))
print("Especificidad:", round(especificidad, 4))
print("Error de mala clasificacion:", round(error, 4))
print("AUC:", round(auc, 4))

fpr, tpr, thresholds = roc_curve(y_test, probabilidades)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Especificidad")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC")
plt.legend()
plt.show()
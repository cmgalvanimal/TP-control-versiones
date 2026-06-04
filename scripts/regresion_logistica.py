from pathlib import Path
import sys
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.metrics import auc

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

from src.carga_datos import CargaDatos

ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

df = cargador.cargar_datos()

y = 1*(df['Gls'] > 0)
"""Creamos un vector booleano que revisa si el jugador metió un gol o no"""
p_sombrero = np.mean(y) #Proporción de todos los jugadores que metieron un gol
print(f'La proporción de jugadores que metieron gol en la temporada es de: {p_sombrero}')


n = len(y)
n_train = int(n*0.8)
n_test = n - n_train
random.seed(10)
indices_train = random.sample(range(n), n_train)
#print(f'Índices seleccionados para train: {indices_train}')
datos_test = df.drop(indices_train)
print(f'Tamaño de datos_test: {len(datos_test)}')
datos_train = df.iloc[indices_train]
print(f'Tamaño de datos_train: {len(datos_train)}')

x_train = (datos_train['Min'])
y_train = 1*(datos_train['Gls'] > 0)
#Visualizamos como están distribuidos los datos
_,ax = plt.subplots()
ax.hist(x_train[y==1],edgecolor='black',color="lightgreen",label="Jugadores que metieron gol")
ax.hist(x_train[y==0],edgecolor='black',color="skyblue",alpha=0.7,label="Jugadores que no metieron gol")
plt.title('Distribución de los minutos jugados')
plt.legend()
#Exploratoriamente, parece haber una correlación entre la cantidad de minutos jugados para predecir si metió gol o no.
X_train = sm.add_constant(x_train)
modelo1 = sm.Logit(y_train,X_train)
resultado1 = modelo1.fit()
print(resultado1.summary())

x_test = (datos_test['Min'])
y_test = 1*(datos_test['Gls'] > 0)
X_test = sm.add_constant(x_test)

prob_est1 = resultado1.predict(X_test)
p_values = np.linspace(0, 1, 100)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad = []
especificidad = []

for p in p_values:
    y_pred = 1 * (prob_est1 >= p)

    a = np.sum((y_pred == 1) & (y_test == 1))  # Verdaderos positivos
    b = np.sum((y_pred == 1) & (y_test == 0))  # Falsos positivos
    c = np.sum((y_pred == 0) & (y_test == 1))  # Falsos negativos
    d = np.sum((y_pred == 0) & (y_test == 0))  # Verdaderos negativos
    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (d+b)

    # Agregar a las listas
    sensibilidad.append(sens)
    especificidad.append(espec)

J = np.array(sensibilidad) + np.array(especificidad) - 1
espec=np.array(especificidad)[J==max(J)][0]
sens=np.array(sensibilidad)[J==max(J)][0]

p = np.array(p_values)[J==max(J)][0]
y_pred = 1 * (prob_est1 >= p)

_,ax_ROC = plt.subplots(1,2)

p1=1-np.array(especificidad)[J==max(J)]
p2=np.array(sensibilidad)[J==max(J)]

ax_ROC[0].plot(1 - np.array(especificidad), sensibilidad)
ax_ROC[0].scatter(p1,p2, color='red')
ax_ROC[0].grid(True)

plt.subplot(1,2,1)
plt.title('curva ROC, Modelo 1')
plt.xlabel('1 - Especificidad')
plt.ylabel('Sensibilidad')

tabla1 =  pd.DataFrame({
    'y_test=1': [np.sum((y_pred == 1) & (y_test == 1)), np.sum((y_pred == 0) & (y_test == 1))],
    'y_test=0': [np.sum((y_pred == 1) & (y_test == 0)), np.sum((y_pred == 0) & (y_test == 0))],

}, index=['y_pred=1', 'y_pred=0'])

print(tabla1)
AUC = auc(1- np.array(especificidad),sensibilidad)

print(f'Modelo 1 \n Sensibilidad {sens:.3f} \n Especificidad {espec:.3f} \n AUC {AUC:.3f} \n Error de mala clasificación total {(tabla1.iloc[0,1] + tabla1.iloc[1,0])/len(y_test):.3f} \n Youden {max(J):.3f} \n Punto p de corte {p:.3f}') 
#El área bajo la curva es de 0.763 por lo que decidimos que el modelo es regular, sin embargo, tiene una especificidad muy baja.

#Veamos ahora que ocurre si incluimos la posición del jugador
dummies_pos = (pd.get_dummies(df['Pos'],drop_first=True))*1
dummies_pos_train = dummies_pos.iloc[indices_train]
dummies_pos_test = dummies_pos.drop(indices_train)

X_train = pd.concat([X_train,dummies_pos_train],axis=1)
X_test = pd.concat([X_test,dummies_pos_test],axis=1)

modelo2 = sm.Logit(y_train,X_train)
resultado2 = modelo2.fit()
print(resultado2.summary())

prob_est2 = resultado2.predict(X_test)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad = []
especificidad = []

for p in p_values:
    y_pred = 1 * (prob_est2 >= p)

    a = np.sum((y_pred == 1) & (y_test == 1))  # Verdaderos positivos
    b = np.sum((y_pred == 1) & (y_test == 0))  # Falsos positivos
    c = np.sum((y_pred == 0) & (y_test == 1))  # Falsos negativos
    d = np.sum((y_pred == 0) & (y_test == 0))  # Verdaderos negativos
    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (d+b)

    # Agregar a las listas
    sensibilidad.append(sens)
    especificidad.append(espec)

J = np.array(sensibilidad) + np.array(especificidad) - 1
espec=np.array(especificidad)[J==max(J)][0]
sens=np.array(sensibilidad)[J==max(J)][0]

p = np.array(p_values)[J==max(J)][0]
y_pred = 1 * (prob_est2 >= p)

p1=1-np.array(especificidad)[J==max(J)]
p2=np.array(sensibilidad)[J==max(J)]

ax_ROC[1].plot(1 - np.array(especificidad), sensibilidad)
ax_ROC[1].scatter(p1,p2, color='red')
ax_ROC[1].grid(True)

plt.subplot(1,2,2)
plt.title('curva ROC, Modelo 2')
plt.xlabel('1 - Especificidad')
plt.ylabel('Sensibilidad')

tabla2 =  pd.DataFrame({
    'y_test=1': [np.sum((y_pred == 1) & (y_test == 1)), np.sum((y_pred == 0) & (y_test == 1))],
    'y_test=0': [np.sum((y_pred == 1) & (y_test == 0)), np.sum((y_pred == 0) & (y_test == 0))],

}, index=['y_pred=1', 'y_pred=0'])

print(tabla2)
AUC = auc(1- np.array(especificidad),sensibilidad)

print(f'Modelo 2 \n Sensibilidad {sens:.3f} \n Especificidad {espec:.3f} \n AUC {AUC:.3f} \n Error de mala clasificación total {(tabla2.iloc[0,1] + tabla2.iloc[1,0])/len(y_test):.3f} \n Youden {max(J):.3f} \n Punto p de corte {p:.3f}') 

plt.show()

'''
Elegimos el modelo 2, ya que es el que mayor AUC tiene, además de que tiene mejor sensibilidad y especificidad, y una menos proporción de error.
El modelo nos dice que:
En función a los minutos jugados y la posición del jugador, varía la probabilidad de que haya metido gol durante la temporada.
'''
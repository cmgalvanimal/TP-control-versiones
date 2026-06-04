import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

datos = pd.read_csv("C:\\Users\\camil\\Downloads\\GitHub\\TP-control-versiones\\data\\jugadores_2024_2025.csv", sep=',')

print(f'Datos:\n{datos.head()}')


# RESOLUCIÓN:

# Queremos responder la siguiente pregunta:
#¿Qué características de un jugador permiten explicar la probabilidad de que anote al menos un gol durante la temporada?


# Para responder a esta pregunta se eligen las siguientes 3 variables predictivas:

# (1) Min:
# Un jugador que juega más tiempo tiene más oportunidades de marcar.

# (2) SoT:
# Cuantas más veces un jugador patee al arco, más oportunidades tiene de marcar.

# (3) SCA:
# Refleja la capacidad ofensiva de un jugador para generar situaciones de remate. Cuanto mayor sea este valor, mayor debería ser la probabilidad de que el jugador convierta al menos un gol.

# ---

# Definimos la variable respuesta:

y = 1 * (datos['Gls'] > 0)
print('\nY =\n', y.head())

# y = 1  -->   El jugador marcó al menos  un gol.
# y = 0  -->   El jugador no marcó ningún gol.

print('\n',y.value_counts())
print('\nHay 1358 jugadores con al menos un gol.')
print('\nHay 1496 jugadores sin ningún gol.')



###             --- PUEBAS DE MODELOS ---


#          ------       Simples          ------

## Prueba 1)

print('\nMinutos Jugados:')

x1 = datos['Min']

X1 = sm.add_constant(x1)
modelo1 = sm.Logit(y, X1)
result1 = modelo1.fit()

print('\nModelo 1: \n',result1.summary())



## Prueba 2)

print('\nTiros al Arco')

x2 = datos['SoT']

X2 = sm.add_constant(x2)
modelo2 = sm.Logit(y, X2)
result2 = modelo2.fit()

print('\nModelo 2: \n',result2.summary())



## Prueba 3)

print('\nAcciones que Generan Tiros')

x3 = datos['SCA']

X3 = sm.add_constant(x3)
modelo3 = sm.Logit(y, X3)
result3 = modelo3.fit()

print('\nModelo 3: \n', result3.summary())




#          ------       Múltiples          ------

## Prueba 4)
x4 = np.column_stack((x1,x2,x3))
X4 = sm.add_constant(x4)

modelo4 = sm.Logit(y, X4)
result4 = modelo4.fit()

print('\nModelo 4: \n', result4.summary())



## Prueba 5)  Sin x1

x5 = np.column_stack((x2,x3))
X5 = sm.add_constant(x5)

modelo5 = sm.Logit(y,X5)
result5 = modelo5.fit()

print('\nModelo 5: \n', result5.summary())



## Prueba 6) Sin x3

x6 = np.column_stack((x1,x2))
X6 = sm.add_constant(x6)

modelo6 = sm.Logit(y, X6)
result6 = modelo6.fit()

print('\nModelo 6: \n', result6.summary())



## Prueba 7) Sin x2

x7 = np.column_stack((x1,x3))
X7 = sm.add_constant(x7)

modelo7 = sm.Logit(y, X7)
result7 = modelo7.fit()

print('\nModelo 7: \n', result7.summary())


###            ----  1° CONCLUSIÓN  ----

# Usando .summary podemos ver que inguno de los modelos múltiples es el adecuado para responder nuestro problema, ya que el p-valor (en todos) es mayor a 0.5
# Por lo tanto, el mejor modelo que podríamos elegir es uno de los 3 primeros. 

# Pero como los 3 primeros son significativamente distintos de cero (están relacionados con la variable respuesta), para elegir entre los 3 es mejor compararlos graficamente. 



###            ----  COMPARACIÓN GRÁFICA DE LOS MODELOS  ---- 


##                     -- Matriz de confusión: --

# MODELO 1)

# Cálculos
aux = np.dot(X1, result1.params)
prob_est = np.exp(aux) / (1 + np.exp(aux))
y_pred1 = 1 * (prob_est >= 0.5)

a = np.sum((y_pred1 == 1) & (y == 1))  # Verdaderos positivos
b = np.sum((y_pred1 == 1) & (y == 0))  # Falsos positivos
c = np.sum((y_pred1 == 0) & (y == 1))  # Falsos negativos
d = np.sum((y_pred1 == 0) & (y == 0))  # Verdaderos negativos

# Tabla de confusión
tabla1 =  pd.DataFrame({
    'y_test=1': [a, c],
    'y_test=0': [b, d],

}, index=['y_pred=1', 'y_pred=0'])

print('\nMatriz de confusión: (modelo 1) \n')

print(tabla1)


# MODELO 2)

# Cálculos
aux = np.dot(X2, result2.params)
prob_est = np.exp(aux) / (1 + np.exp(aux))
y_pred2 = 1 * (prob_est >= 0.5)

a2 = np.sum((y_pred2 == 1) & (y == 1))  # Verdaderos positivos
b2 = np.sum((y_pred2 == 1) & (y == 0))  # Falsos positivos
c2 = np.sum((y_pred2 == 0) & (y == 1))  # Falsos negativos
d2 = np.sum((y_pred2 == 0) & (y == 0))  # Verdaderos negativos

# Tabla de confusión
tabla2 =  pd.DataFrame({
    'y_test=1': [a2, c2],
    'y_test=0': [b2, d2],

}, index=['y_pred=1', 'y_pred=0'])

print('\nMatriz de confusión: (modelo 2) \n')

print(tabla2)


# MODELO 3)

# Cálculos
aux = np.dot(X3, result3.params)
prob_est = np.exp(aux) / (1 + np.exp(aux))
y_pred3 = 1 * (prob_est >= 0.5)

a3 = np.sum((y_pred3 == 1) & (y == 1))  # Verdaderos positivos
b3 = np.sum((y_pred3 == 1) & (y == 0))  # Falsos positivos
c3 = np.sum((y_pred3 == 0) & (y == 1))  # Falsos negativos
d3 = np.sum((y_pred3 == 0) & (y == 0))  # Verdaderos negativos

# Tabla de confusión
tabla3 =  pd.DataFrame({
    'y_test=1': [a3, c3],
    'y_test=0': [b3, d3],

}, index=['y_pred=1', 'y_pred=0'])

print('\nMatriz de confusión: (modelo 3) \n')

print(tabla3)



##                     -- Proporciciones: --


# MODELO 1)

# Calcular sensibilidad y especificidad
sens1 = a / (a+c)
espec1 = d / (d+b)

print('\nEn el modelo 1:')
print(f'\nLa sensibilidad es {sens1}, y la especificidad es {espec1}')

# Calcular los falsos:
FPR_1 = 1 - espec1 # Falsos positivos
FNR_1 = 1 - sens1 # Falsos negativos

print(f'\nLos falsos positivos son {FPR_1}, y y los falsos negativos son {FNR_1}')



# MODELO 2)

# Calcular sensibilidad y especificidad
sens2 = a2 / (a2+c2)
espec2 = d2 / (d2+b2)

print('\nEn el modelo 2:')
print(f'\nLa sensibilidad es {sens2}, y la especificidad es {espec2}')

# Calcular los falsos:
FPR_2 = 1 - espec2 # Falsos positivos
FNR_2 = 1 - sens2 # Falsos negativos

print(f'\nLos falsos positivos son {FPR_2}, y y los falsos negativos son {FNR_2}')



# MODELO 3)

# Calcular sensibilidad y especificidad
sens3 = a3 / (a3+c3)
espec3 = d3 / (d3+b3)

print('\nEn el modelo 3:')
print(f'\nLa sensibilidad es {sens3}, y la especificidad es {espec3}')

# Calcular los falsos:
FPR_3 = 1 - espec3 # Falsos positivos
FNR_3 = 1 - sens3 # Falsos negativos

print(f'\nLos falsos positivos son {FPR_3}, y y los falsos negativos son {FNR_3}')





##                     -- Curva ROC: --

# MODELO 1)

# Generar valores de p
p_values = np.linspace(0, 1, 1000)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad = []
especificidad = []

for p in p_values:
    y_pred1 = 1 * (prob_est >= p)

    a = np.sum((y_pred1 == 1) & (y == 1))  # Verdaderos positivos
    b = np.sum((y_pred1 == 1) & (y == 0))  # Falsos positivos
    c = np.sum((y_pred1 == 0) & (y == 1))  # Falsos negativos
    d = np.sum((y_pred1 == 0) & (y == 0))  # Verdaderos negativos

    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (d+b)

    # Agregar a las listas
    sensibilidad.append(sens)
    especificidad.append(espec)

# Gráfico
indice_0_5 = np.where(p_values == 0.5)[0]
p1= 1-np.array(especificidad)[indice_0_5]
p2= np.array(sensibilidad)[indice_0_5]

plt.figure(figsize=(10, 5))
plt.plot(1 - np.array(especificidad), np.array(sensibilidad))

plt.scatter(p1,p2, color='red')

plt.xlabel('1-Especificidad')
plt.ylabel('Sensibilidad')
plt.title('curva ROC (modelo 1)')
plt.grid(True)
plt.show()



# MODELO 2)

# Generar valores de p
p_values = np.linspace(0, 1, 1000)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad2 = []
especificidad2 = []

for p in p_values:
    y_pred2 = 1 * (prob_est >= p)

    a = np.sum((y_pred2 == 1) & (y == 1))  # Verdaderos positivos
    b = np.sum((y_pred2 == 1) & (y == 0))  # Falsos positivos
    c = np.sum((y_pred2 == 0) & (y == 1))  # Falsos negativos
    d = np.sum((y_pred2 == 0) & (y == 0))  # Verdaderos negativos

    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (d+b)

    # Agregar a las listas
    sensibilidad2.append(sens)
    especificidad2.append(espec)

# Gráfico
indice_0_5 = np.where(p_values == 0.5)[0]
p1= 1-np.array(especificidad2)[indice_0_5]
p2= np.array(sensibilidad2)[indice_0_5]

plt.figure(figsize=(10, 5))
plt.plot(1 - np.array(especificidad2), np.array(sensibilidad2))

plt.scatter(p1,p2, color='red')

plt.xlabel('1-Especificidad')
plt.ylabel('Sensibilidad')
plt.title('curva ROC (modelo 2)')
plt.grid(True)
plt.show()



# MODELO 3)

# Generar valores de p
p_values = np.linspace(0, 1, 1000)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad3 = []
especificidad3 = []

for p in p_values:
    y_pred3 = 1 * (prob_est >= p)

    a = np.sum((y_pred3 == 1) & (y == 1))  # Verdaderos positivos
    b = np.sum((y_pred3 == 1) & (y == 0))  # Falsos positivos
    c = np.sum((y_pred3 == 0) & (y == 1))  # Falsos negativos
    d = np.sum((y_pred3 == 0) & (y == 0))  # Verdaderos negativos

    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (d+b)

    # Agregar a las listas
    sensibilidad3.append(sens)
    especificidad3.append(espec)

# Gráfico
indice_0_5 = np.where(p_values == 0.5)[0]
p1= 1-np.array(especificidad3)[indice_0_5]
p2= np.array(sensibilidad3)[indice_0_5]

plt.figure(figsize=(10, 5))
plt.plot(1 - np.array(especificidad3), np.array(sensibilidad3))

plt.scatter(p1,p2, color='red')

plt.xlabel('1-Especificidad')
plt.ylabel('Sensibilidad')
plt.title('curva ROC (modelo 3)')
plt.grid(True)
plt.show()




##                     -- AUC: --

# Área bajo la curva

# MODELO 1)
roc_auc1 = auc(1 - np.array(especificidad), sensibilidad)
print("\nAUC (modelo 1):", roc_auc1)

# MODELO 2)
roc_auc2 = auc(1 - np.array(especificidad2), sensibilidad2)
print("\nAUC (modelo 2):", roc_auc2)

# MODELO 3)
roc_auc3 = auc(1 - np.array(especificidad3), sensibilidad3)
print("\nAUC (modelo 3):", roc_auc3)




###            ----  2° CONCLUSIÓN  ----

sensi = np.array([sens1, sens2, sens3])
print(f'\nLa sensibilidad más alta es {max(sensi)}, y pertenece al modelo 2.')

esp = np.array([espec1, espec2, espec3])
print(f'\nLa especificidad más alta es {max(esp)}, y pertenece al modelo 2.')

# Sabemos que, cuanto más alta la sensibilidad y escificidad, más efectivo es nuestro modelo. En este caso, el modelo con sensibilidad y escificidad más altas es el modelo 2.

# Entonces, el modelo más efectivo y que mejor se adapta para responder nuestro problema es: el MODELO 2 (Goles vs. Tiros al Arco).

# Por lo tanto: Cuantas más veces un jugador patee al arco, más oportunidades tiene de marcar al menos un gol.


import pandas as pd  # Importa pandas como pd
from sklearn.cross_validation import train_test_split  # De scikit-learn.cross_validation se importa train_test_split
from sklearn.linear_model import LinearRegression   # De scikit-learn.linear_model se importa LinearRegression
import matplotlib.pyplot as plt    #Se importa matplotlib como plt 

dataset = pd.read_csv('Salary_Data.csv') # Se lee el dataset "Salary_Data.csv"
X = dataset.iloc[:,:-1].values # Se selecciona la primera columna y se asigna en la variable X
y = dataset.iloc[:,1].values # Se selecciona la segunda columna y se asigna a la variable y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) #Se crean subgrupos de datos para entrenamiento
regressor = LinearRegression() # Se asigna la funcion LinearRegression a la variable regressor
regressor.fit(X_train, y_train) #  Se crea un modelo de regresion lineal utilizando los datos de entrenamiento 

y_pred = regressor.predict(X_test) #Se asigna a y_pred el modelo de regresion regressor.predict de losdatos en X_test
plt.scatter(X_train, y_train, color = 'red') # Se graficaran los datos de X_train con respecto a los datos de y_train y se muestran en color rojo
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # En la grafica se muestran los datos predictivos de X_train en color azul
plt.show() #muestra en pantalla la grafica scatter en conjunto con plot.

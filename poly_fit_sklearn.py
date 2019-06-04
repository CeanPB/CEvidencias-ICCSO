from sklearn.linear_model import LinearRegression # De scikit-learn.linear_model se importa LinearRegression
from sklearn.preprocessing import PolynomialFeatures #De scikit_learn.preprocessing se importa PolynomialFeatures que nos permitira determinar un grado de evaluacion
import matplotlib.pyplot as plt #Se importa matplotlib como plt
import pandas as pd #Se importa pandas como pd

dataset = pd.read_csv('exam_B_dataset.csv') #Se asigna a dataset los datos que se encuentran en el archivo exam_B_dataset.csv
X = dataset.iloc[:,:-1].values #Se asigna la primer columna del dataset a X
y = dataset.iloc[:,1].values #Se asignan los valores de la segunda columna del dataset a y

lin_reg = LinearRegression() #Se asigna la funcion Linear Regression a la variable lin_reg
poly_reg = PolynomialFeatures(degree=4) #Se asigna la funcion PolynomialFeatures en este caso de grado 4 a la variable poly_reg

X_poly = poly_reg.fit_transform(X) #Se asigna la funcion fit_transform de X es decir la funcion que evalua la regresion polinomica de X a X_poly
poly_reg.fit(X_poly,y) # Se ajusta la regresion polinomial al conjunto de datos
lin_reg.fit(X_poly,y) # Se ajusta la regresion polinomial al conjunto de datos


plt.scatter(X,y, color ='red') #Se visualiza mediante graficas el resultado de la regresion polinomial
plt.scatter(X,lin_reg.predict(poly_reg.fit_transform(X)), color = 'black')
plt.show()

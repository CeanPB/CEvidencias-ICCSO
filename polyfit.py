
import pandas as pd #Se importa pandas como pd
import matplotlib.pyplot as plt #Se importa pyplot de matplotlib como plt
import numpy as np #se importa numpy como np
import sympy as S #se importa sympy como S

file = 'exam_B_dataset.csv' #Se elige el archivo de datos y se asigna a file y se asignan a sus respectivos campos X o Y
data = np.loadtxt(file, delimiter = ',', skiprows = 0, usecols=[0,1])
X = np.loadtxt(file, delimiter= ',' ,skiprows = 0, usecols= [0])
Y = np.loadtxt(file, delimiter= ',' ,skiprows = 0, usecols= [1])



def polyfit2(x,y,n): #Se define polyfit2, es la regresiion polinimial de grado 2

    def inv(A): #Se define la inversa de A
        return np.linalg.inv(A)
    def trans(A): #Se define la transpuesta de A
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B): #Se define el producto de A por B
        return np.dot(A,B)

    # Se retoman los valores necesarios de nuestro dataset, para realizar las operaciones posteriores y se asignan a variables.
    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    A=np.hstack([c1,c2,c3])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit2(X,Y,2)) #Se realiza la operacion logica de nuestra ecuacion

#Se asignan simbolos a las variables x y y para idenificarlos mejor
x = S.symbols('x') 
y = S.symbols('y')


y = -3.33 + 0.53 * x -1.94 * pow(x,2) +0.523 * pow(x,3) +0.496 * pow(x,2) #Estos valores son los generados por la parte superior y los utilizamos para la posterior graficacion
f = S.lambdify(x,y,'math')
yen = f(X)
print(yen)
yout= yen.astype(list)

#Graficamos nuestros resultados comparando los datos del dataset original con los datos obtenidos por la regresion polinomial de grado 2
plt.scatter(X,Y, color = 'orange')
plt.scatter(X,yout , color='purple')
plt.show()





def polyfit3(x,y,n): #Se define polyfit3, es la regresiion polinimial de grado 3

    def inv(A): # Definimos la inversa de A
        return np.linalg.inv(A)
    def trans(A):# Definimos la transpuesta de A
        return A.getT()
    def oneMat(xl,n): 
        return np.ones((xl,n),dtype=int)
    def prod(A,B): #Definimos el producto de A por B
        return np.dot(A,B)

    # Se retoman los valores necesarios de nuestro dataset, para realizar las operaciones posteriores y se asignan a variables.
    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4 = np.power(c2,3) #Se agrega una nueva columna para cumplir con los grados que deseamos en este caso 3, la primer columna es de unos
                         # la seguna es de x, la tercera de x cuadrada y la cuarta de x cubica.

    A=np.hstack([c1,c2,c3,c4])


    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit3(X,Y,3)) #Se realiza la operacion logica de nuestra ecuacion

x = S.symbols('x')
y = S.symbols('y')


y = -3.33 + 0.53 * x -1.94 * pow(x,2) +0.523 * pow(x,3) +0.496 * pow(x,3)  #Estos valores son los generados por la parte superior y los utilizamos para la posterior graficacion
f = S.lambdify(x,y,'math')
yen = f(X)
print(yen)
yout= yen.astype(list)

#Graficamos nuestros resultados comparando los datos del dataset original con los datos obtenidos por la regresion polinomial de grado 3
plt.scatter(X,Y, color = 'orange')
plt.scatter(X,yout , color='purple')
plt.show()





def polyfit4(x,y,n): #Se define polyfit3, es la regresiion polinimial de grado 4

    def inv(A): # Definimos la inversa de A
        return np.linalg.inv(A)
    def trans(A):# Definimos la transpuesta de A
        return A.getT()
    def oneMat(xl,n): 
        return np.ones((xl,n),dtype=int)
    def prod(A,B): #Definimos el producto de A por B
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4 = np.power(c2,3)
    c5= np.power(c2,4) #Se agrega una nueva columna para cumplir con los grados que deseamos en este caso 4, la primer columna es de unos
                         # la seguna es de x, la tercera de x cuadrada, la cuarta de x cubica y la quinta es x a la cuarta.

    A=np.hstack([c1,c2,c3,c4,c5])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit4(X,Y,4))#Se realiza la operacion logica de nuestra ecuacion

x = S.symbols('x')
y = S.symbols('y')


y = -3.33 + 0.53 * x -1.94 * pow(x,2) +0.523 * pow(x,3) +0.496 * pow(x,4) #Estos valores son los generados por la parte superior y los utilizamos para la posterior graficacion
f = S.lambdify(x,y,'math')
yen = f(X)
print(yen)
yout= yen.astype(list)

#Graficamos nuestros resultados comparando los datos del dataset original con los datos obtenidos por la regresion polinomial de grado 4
plt.scatter(X,Y, color = 'orange')
plt.scatter(X,yout , color='purple')
plt.show()

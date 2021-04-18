# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 19:43:17 2021

@author: BERETE Mohamed Loua
"""

from data import *
from GradDesc import *
import numpy as np
import matplotlib.pyplot as plt


#Importation des données d'entrée x et de sortie y
x = data['input']
y = data["target"]

print(x.shape)

y = y.reshape(y.shape[0],1)
print(y.shape)

#Création d'une fonction d'encodage one Hot

def One_hot(y):
    y_temp = np.zeros((90,4))
    for i in range(y.shape[0]):
        y_temp[i, int(y[i])] =  1
        
    return y_temp

print(One_hot(y))
print((One_hot(y)).shape)

def softmax(v):
    return(((np.exp(v).T)/(np.exp(v).sum(axis = 1))).T)


# Création du programme de calcul du nombre d'erreur :

def nbr_error(u,S):
    In = S[:,:-1]
    Out = S[:,-1]
    
    u = u.reshape(4,4)
    x = u[:,:-1]
    b = u[:,-1]
    
    vect = In.dot(x.T) + b
    vect = softmax(vect)
    
    
    y_prediction =  np.array(vect.argmax(axis=1))
    
    
    return (Out != y_prediction).sum()

# Création de la fonction de coût avec la log vraisemblance et la fonction softmax

def loss(u,S):
    In = S[:,:-1]
    Out = S[:,-1]
    
    u = u.reshape(4,4)
    x = u[:,:-1]
    b = u[:,-1]
    
    vect = In.dot(x.T) + b
    vect = softmax(vect)
    y = One_hot(Out)
    y_temp =  (y * vect).sum(axis=1)
    
    loss = -np.log(y_temp).sum()
    return loss


u = np.random.randn(16)
S = np.column_stack((x,y))

print(loss(u,S))
print(nbr_error(u,S))



grad = grad_desc_n(loss, S, 16, 200, step = 0.005, x_0 = None)

u_solution = grad

print(loss(u_solution,S))
print(nbr_error(u_solution,S))

print("la précision de mon classifieur aléatoire est de", ((y.shape[0] - nbr_error(u,S))/y.shape[0])*100,"%" )
print("Le programme à fait %d erreurs de classification"%(nbr_error(u,S)))

#Valeur d'un U qui me permettait d'avoir uniquement 5 erreur
#u_5 = [ 1.75470826  0.82757108 -2.41863613  1.5615079   0.36667595 -2.55537334
 # 2.32947521  1.67212793 -2.2924324   3.09969007  0.58678593 -1.06276498
  #0.86385175  0.41507802  1.04728817 -1.79141909]

print("ma solution est \n", u_solution)

print("la précision de mon classifieur optimisé est de", ((y.shape[0] - nbr_error(u_solution,S))/y.shape[0])*100,"%" )
print("Le programme à fait %d erreurs de classification"%(nbr_error(u_solution,S)))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:53:10 2022

@author: mo
"""

# Import Nessary Libraries
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, exp, pi, sqrt

#Learning Rate 
alpha = 0.0027

#Initial Point
X = np.array([0.071, -0.2, 0.06])

#Epsilon value
epsilon = 10**-12

#Original Function ---> F(x1,x2,x3) = 0.5xg1^2 + 0.5xg2^2 + 0.5xg3^2
def F(X):
    g1 = (3*X[0]) - cos(X[1] * X[2]) - 0.5
    g2 = (X[0]**2) - (81*(X[1] + 0.1)**2) + sin(X[2]) + 1.06
    g3 = exp(-X[0] * X[1]) + (20*X[2]) + ((10*pi - 3)/3)
    
    F = (0.5*g1**2) + (0.5*g2**2) + (0.5*g3**2)

    return g1,g2,g3,F

#Gradient Vector of F ---> dF = [dx1, dx2, dx3]
def Gradient_F(X):
    
    F_val = F(X)
    
    G = [(3*F_val[0]) + (2*X[0]*F_val[1]) - ((X[1]*F_val[2])*(exp(-(X[0] * X[1])))),
      (X[2]*F_val[0]*sin(X[2] * X[1])) - (162*F_val[1]*(X[1] + 0.1)) - (F_val[2]*X[0]*(exp(-(X[0] * X[1])))),
      (F_val[0]*X[1]*sin(X[2] * X[1])) + (F_val[1]*cos(X[2])) + (20*F_val[2])]
    G = np.array(G)
    
    GM = Gradient_Mag(G)
    return G,GM,F_val[3]

#Gradient Magnitude ----> sqrt(dx1^2 + dx2^2 + dx3^2)
def Gradient_Mag(G):
    grad_magnitude = sqrt((G[0]**2) + (G[1]**2) + (G[2]**2))
    return grad_magnitude

#Define Lists to store the results and plot them
Grad_Output = []
Func_Output = []
It = []

for i in range (0,100):
    ## Calculate gradient vector and magnitude
    Grad = Gradient_F(X)
    print("GM:",Grad[1])
    
    Grad_Output.append(Grad[1])
    Func_Output.append(Grad[2])
    It.append(i)
    
    ## check if the GM < epsilon 
    if Grad[1] < epsilon:
        break
    else:
    ## calculate X_i+1 = X_i - (alpha*G)     
        X = X - (alpha * Grad[0])
        print("#It:",i+1," X1_new:",X[0]," X2_new:",X[1]," X3_new:",X[2])
                
    ## repeat

#Visualize Results
print("X1_f= ",X[0]," X2_f= ",X[1]," X3_f= ",X[2])
plt.plot(It,Grad_Output,label="Gradient")
plt.plot(It,Func_Output,label="Function")
plt.legend()
plt.show()
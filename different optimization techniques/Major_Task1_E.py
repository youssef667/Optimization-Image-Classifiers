#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:57:46 2022

@author: mo
"""

# Import Nessary Libraries
import numpy as np
from scipy.optimize import brent as br
import matplotlib.pyplot as plt
from math import cos, sin, exp, pi, sqrt

#Initial Point
X0 = np.array([0.0, 0.0, 0.0])

#Epsilon value
epsilon = 10**-12

def g1(X):
    return (3*X[0]) - cos(X[1] * X[2]) - 0.5

def g2(X):
    return (X[0]**2) - (81*(X[1] + 0.1)**2) + sin(X[2]) + 1.06

def g3(X):
    return exp(-X[0] * X[1]) + (20*X[2]) + ((10*pi - 3)/3)

def F(X):
    return (0.5*g1(X)**2) + (0.5*g2(X)**2) + (0.5*g3(X)**2)

#Gradient Vector of F ---> dF = [dx1, dx2, dx3]
def Gradient_F(X):
    G = [(3*g1(X)) + (2*X[0]*g2(X)) - (X[1]*g3(X)*exp(-(X[0] * X[1]))),
         (X[2]*g1(X)*sin(X[2] * X[1])) - (162*g2(X)*(X[1] + 0.1)) - (g3(X)*X[0]*exp(-(X[0] * X[1]))),
         (g1(X)*X[1]*sin(X[2] * X[1])) + (g2(X)*cos(X[2])) + (20*g3(X))]
    return np.array(G)         

#Gradient Magnitude ----> sqrt(dx1^2 + dx2^2 + dx3^2)
def Gradient_Mag(G):
    return sqrt((G[0]**2) + (G[1]**2) + (G[2]**2))

#Gradient Initial Value 
Direction = np.array(Gradient_F(X0))
GM = Gradient_Mag(Gradient_F(X0))

X = X0
alpha_opt = 0.0
F_new = 0.0
count = 0

#Define Lists to store the results and plot them
GradMag = []
Func_Output = []
It = []

while GM > epsilon:
    
    count = count + 1
    It.append(count)
    
    def phi(alpha):
        return F(X - alpha*Gradient_F(X))
    
    alpha_opt = br(phi)
    X = X - alpha_opt*Gradient_F(X)
    
    GM = Gradient_Mag(Gradient_F(X))
    GradMag.append(GM)
    
    F_new = F(X)
    Func_Output.append(F_new)
    
    print("It:",count,"-->" ,"Alpha = ", alpha_opt, "X = ", X, "GM = ", GM, "F(X) = ", F_new)

print('#Iterations = ', count)
print("Alpha_opt = ",alpha_opt)
print("Grad_Mag = ",GM)
print("X_min = ", X)

#Visualize Results
plt.plot(It,GradMag,label="Gradient")
plt.plot(It,Func_Output,label="Function")
plt.legend()
plt.show()   
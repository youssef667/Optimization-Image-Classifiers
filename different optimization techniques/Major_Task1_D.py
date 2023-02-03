#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:57:12 2022

@author: mo
"""

# Import Nessary Libraries
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, exp, pi, sqrt

#Initial Point
X = [0.0, 0.0, 0.0]
X = np.array(X)

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
    return G,GM,F_val[3],F_val[0],F_val[1],F_val[2]

#Gradient Magnitude ----> sqrt(dx1^2 + dx2^2 + dx3^2)
def Gradient_Mag(G):
    grad_magnitude = sqrt((G[0]**2) + (G[1]**2) + (G[2]**2))
    return grad_magnitude

#Hessien Matrix
def Hessien_F(X,g):

    dx11 = 9 + 2*g[1] + 4*X[0]**2 + (X[1]**2)*(exp(-X[0] * X[1]))*(g[2]*exp(-X[0] * X[1]))    
    dx22 = (X[2]**2)*( (g[0]*cos(X[1]*X[2])) + (sin(X[1]*X[2])**2) ) - 162*g[1] + 26244*(X[1]+0.1)**2 + (X[0]**2)*(exp(-X[0] * X[1]))*(g[2]*exp(-X[0] * X[1]))
    dx33 = (X[1]**2)*( (g[0]*cos(X[1]*X[2])) + (sin(X[1]*X[2])**2) ) - g[1]*sin(X[2]) + (cos(X[2])**2) + 400
    
    dx12 = 3*X[2]*sin(X[1]*X[2]) - 324*X[0]*(X[1]+0.1) + (X[0]*X[1])*(exp(-X[0] * X[1]))*(g[2]*exp(-X[0] * X[1]))
    dx21 = dx12
    
    dx23 = g[0]*(sin(X[1]*X[2])+(X[1]*X[2]*sin(X[1]*X[2])**2)) + cos(X[2]*(-162*(X[1]+0.1))) - 20*X[0]*exp(-X[0] * X[1])
    dx32 = dx23
    
    dx13 = 3*X[1]*sin(X[1]*X[2]) + 2*X[0]*cos(X[2]) - 20*X[1]*exp(-X[0] * X[1])
    dx31 = dx13
    
    H = [[dx11,dx21,dx31],
         [dx12,dx22,dx32],
         [dx13,dx23,dx33]] 
    return H

#Calculate Alpha
def Calc_Alpha(H):
    alpha = np.linalg.inv(H)
    return alpha
    
#Define Lists to store the results and plot them
Grad_Output = []
Func_Output = []
It = []
g = np.zeros(shape=(3))

for i in range (0,100):
    ## Calculate gradient vector and magnitude
    Grad = Gradient_F(X)
    print("GM:",Grad[1])
    
    g[0] = Grad[3]
    g[1] = Grad[4]
    g[2] = Grad[5]
    
    Grad_Output.append(Grad[1])
    Func_Output.append(Grad[2])
    It.append(i)
    
    ## check if the GM < epsilon 
    if Grad[1] < epsilon:
        break
    else:
    ## calculate X_i+1 = X_i - (alpha*G) && alpha = inv(H)
        Hess = Hessien_F(X, g)
        alpha = Calc_Alpha(Hess)
        print("Alpha:",alpha)
        
        X = X - (np.dot(alpha, Grad[0]))
        print("#It:",i+1," X1_new:",X[0]," X2_new:",X[1]," X3_new:",X[2])
                
    ## repeat

#Visualize Results
print("X1_f= ",X[0]," X2_f= ",X[1]," X3_f= ",X[2])
plt.plot(It,Grad_Output,label="Gradient")
plt.plot(It,Func_Output,label="Function")
plt.legend()
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:53:13 2022

@author: mo
"""
# Import Nessary Libraries
import numpy as np
from math import cos, sin, exp, pi
from sympy import * 

x1, x2, x3 = symbols('x1 x2 x3')
g1 = (3*x1) - cos(x2 * x3) - 0.5
g2 = (x1**2) - (81*(x2 + 0.1)**2) + sin(x3) + 1.06
g3 = exp(-x1 * x2) + (20*x3) + ((10*pi - 3)/3)
f = (0.5*(g1**2)) + (0.5*(g2**2)) + (0.5*(g3**2))

#Gradient Vector of F ---> dF = [F1, F2, F3].Transpose        
F1 = diff(f,x1)
F2 = diff(f,x2)
F3 = diff(f,x3)

Gradient = np.array([[F1], [F2], [F3]])
print("---------------------------------------------------------")
print("Gradient = ",Gradient)
print("---------------------------------------------------------")

#Hessian Matrix ---> H = [F11,F21,F31],[F12,F22,F32],[F13,F23,F33]] 
F11 = diff(F1,x1)
F12 = diff(F1,x2)
F13 = diff(F1,x3)

F21 = diff(F2,x1)
F22 = diff(F2,x2)
F23 = diff(F2,x3)

F31 = diff(F3,x1)
F32 = diff(F3,x2)
F33 = diff(F3,x3)

H = np.array([[F11,F21,F31], [F12,F22,F32], [F13,F23,F33]])

print("Hessian = ",H)
print("---------------------------------------------------------")
print("Gradient Shape = ",Gradient.shape)
print("Hessian Shape = ",H.shape)
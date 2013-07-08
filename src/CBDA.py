import numpy as np 
import math 
from random import *

data = np.loadtxt("../input/synthetic_data.dat")

X2 = data[:,0]
Y2 = data[:,1]

#print X[0:100]


# This function search for the K closest neighbours 

# k = Number of Neighbors, D = Dimension of the space we assume here X = Y | CARTESIAN COORDINATES
def Neighbours_Cartesian(k, Res):
    global d_k
    d_k = []
    Fx = np.linspace(np.amin(X2), np.amax(X2), Res)
    Fy = np.linspace(np.amin(Y2), np.amax(Y2), Res)
    for i in Fx:
        for j in Fy:
            d = np.sqrt((X2-i)**2 + (Y2-j)**2)
            d2 = sorted(d)
            d3 = d2[0:k]
            d_k.append(d3)
    #print len(d4)
    #print d4[0:10]
    
Neighbours_Cartesian(12, 200)

def solution(K, Res):
    k = range(1, 100)
    global d_0
    d_0 = []
    for i in range(len(d_k)): #escala los puntos del espacio
        T = []
        for j in range (K):# j escala los vecinos
            teo = d_k[i][j]**2 / k[j] 
            T.append(teo)
        T2 = (np.sqrt(sum(T))) 
        d_0.append(1.0/(T2*T2*np.pi)) #This is divided in order to get n_0
solution(12, 200)

#Sigma Estimator
def sigma_estimator(K, Res):
    k = range(1, 100)
    global sigma_2
    sigma_2 = []
    #NDots = linspace(0, D, Res)
    #ndots = len(NDots)*len(NDots)
    for i in range(len(d_k)): #i escala los puntos del espacio
        S = []
        for j in range (K):# j escala los vecinos 
            sigma1 = (-3*d_k[i][j]**2 + k[j]*d_0[i]**2)
            S.append(sigma1)
        S2 = ((sum(S))) 
        sigma = (-K*d_0[i]**4)/(2*S2)
        sigma_2.append(np.pi*sigma**2) 
    #p  rint type(sigma), dtype(S[1]), type(S), dtype(S2), type(sigma_2)
    
sigma_estimator(12, 200)

# Check color this creates the grid in order to make the density plot

def plots(Res):
    f = open('../output/DensityData.txt', 'w')
    global x
    global y
    x = []
    y = []
    Fx = np.linspace(np.amin(X2) , np.amax(X2)  , Res)
    Fy = np.linspace(np.amin(Y2) , np.amax(Y2)  , Res)
    #FRx = linspace(amin(R*cos(theta)) - 5, amax(R*cos(theta)) + 5 , Res)
    #FRy = linspace(amin(R*sin(theta)) - 5, amax(R*sin(theta)) + 5 , Res)
    for i in Fx:
        for j in Fy:
            x.append(i)
            y.append(j)
    for i in range(len(x)):
        f.write(str(x[i]) + "  " + str(y[i]) + "  " + str(d_0[i]) +"\n") 
    f.close()
    #print len(x), len(y), len(T3)

plots(200)


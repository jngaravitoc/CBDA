import numpy as np 
from math import* 
from random import *
import scipy as sc
import sys

def Neighbours_Cartesian(K, Res, xmin_limit, xmax_limit, ymin_limit, ymax_limit, zmin_limit, zmax_limit):
    """
    Returns the distance to the kth neighboor d_k

    TODO:
    -----
    Use Scikit-learn!
    """
    d_k = []
    if D == 1:
        X = data[:, N_X]
        Fx = np.linspace(xmin_limit, xmax_limit, Res)
        for i in Fx:
            d =  abs(X-i)
            d2 = sorted(d)
            d3 = d2[0:K]
            d_k.append(d3)
    elif D == 2:
        X = data[:, N_X]
        Y = data[:, N_Y]
        Fx = np.linspace(xmin_limit, xmax_limit, Res)
        Fy = np.linspace(ymin_limit, ymax_limit, Res)
        for i in Fx:
            for j in Fy:
                d = np.sqrt((X-i)**2 + (Y-j)**2)
                d2 = sorted(d)
                d3 = d2[0:K]
                d_k.append(d3)
    elif D ==3:
        X = data[:, N_X]
        Y = data[:, N_Y]
        Z = data[:, N_Z]
        Fx = np.linspace(xmin_limit, xmax_limit, Res)
        Fy = np.linspace(ymin_limit, ymax_limit, Res)	
        Fz = np.linspace(zmin_limit, zmax_limit, Res)
        for i in Fx:
            for j in Fy:
                for k3 in Fz:
                    d = np.sqrt((X-i)**2 + (Y-j)**2	+ (Z-k3)**2)
                    d2 = sorted(d)
                    d3 = d2[0:K]
                    d_k.append(d3)
     else:
        print('No aveilable dimension')
        print('Completed neighbours finder')
    #print d4[0:10]
    return d_k



def solution(K):
    k = range(1, 100)
    global d_0
    d_0 = []
    if D == 1:
        for i in range(len(d_k)):
            T = []
            for j in range(K):
                teo = d_k[i][j]/k[j]
                T.append(teo)
                T2 = sum(T)
                d_0.append(K/(T2*T2*np.pi))
    elif D == 2:
        for i in range(len(d_k)): #escala los puntos del espacio
            T = []
            for j in range (K):# j escala los vecinos
                teo = d_k[i][j]**2 / k[j] 
                T.append(teo)
                T2 = np.sqrt(sum(T)) 
                d_0.append(K/(T2*T2*np.pi)) #This is divided in order to get n_0
    elif D ==3: 
        for i in range(len(d_k)):
            T=[]
            for j in range (K): 
                teo = d_k[i][j]**3/ k[j]
                T.append(teo)
                T2 = (sum(T))**(1/3)
                 d_0.append(3.0*K/(4.0*np.pi*T2*T2*T2))
    else:
        print('No available dimension')
        print('Completed density estimation')



def sigma_estimator(K):
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
    print('Completed Sigma estimation')
    print('Writting data in (../output/)')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('please provide a script with the following information')
    else:
        x = sc.genfromtxt(sys.argv[1], dtype='S')
        dic = {}
        for i in range(len(x[:,0])): 
            (key, val) = (x[i,0], x[i, 2])
            dic[(key)] = val

    data = np.loadtxt(dic['data_input'])
    data_output = dic['data_output']
    D = float(dic['D'])
    Res = float(dic['Res'])
    K = int(dic['K'])
    N_X = float(dic['N_X'])
    N_Y = float(dic['N_Y']) 
    N_Z = float(dic['N_Z'])
    xmin_limit = float(dic['xmin_limit'])
    xmax_limit = float(dic['xmax_limit'])
    ymin_limit = float(dic['ymin_limit'])
    ymax_limit = float(dic['ymax_limit'])
    zmin_limit = float(dic['zmin_limit'])
    zmax_limit = float(dic['zmax_limit'])

    print xmin_limit
    print xmax_limit
    print ymin_limit
    print ymax_limit
    print zmin_limit
    print zmax_limit
    # This function search for the K closest neighbours 

    Neighbours_Cartesian(K, Res, xmin_limit, xmax_limit, ymin_limit, ymax_limit, zmin_limit, zmax_limit)
    solution(K)
    sigma_estimator(K)
    # k = Number of Neighbors, D = Dimension of the space we assume here X = Y | CARTESIAN COORDINATES
    #plots(Res, xmin_limit, xmax_limit, ymin_limit, ymax_limit, zmin_limit, zmax_limit)

import numpy as np 
from math import* 
from random import *
import scipy as sc
import sys


if len(sys.argv) < 2:
	print 'please provide a script with the following information'
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

# This function search for the K closest neighbours 

# k = Number of Neighbors, D = Dimension of the space we assume here X = Y | CARTESIAN COORDINATES

def Neighbours_Cartesian(K, Res):
	global d_k
	d_k = [] 
	if D == 1:
		X = data[:, N_X]/15.0
		Fx = np.linspace(np.amin(X), np.amax(X), Res)
		for i in Fx:
			d = (X-i)
			d2 = sorted(d)
			d3 = d2[0:K]
			d_k.append(d3)
	elif D == 2:
		X = data[:, N_X]/15.0
		Y = data[:, N_Y]
    		Fx = np.linspace(np.amin(X), np.amax(X), Res)
    		Fy = np.linspace(np.amin(Y), np.amax(Y), Res)
		for i in Fx:
        		for j in Fy:
            			d = np.sqrt((X-i)**2 + (Y-j)**2)
            			d2 = sorted(d)
           			d3 = d2[0:K]
         			d_k.append(d3)
	elif D ==3:
		X = data[:, N_X]/15.0
		Y = data[:, N_Y]
		Z = data[:, N_Z]
    		Fx = np.linspace(np.amin(X), np.amax(X), Res)
    		Fy = np.linspace(np.amin(Y), np.amax(Y), Res)	
		Fz = np.linspace(np.amin(Z), np.amax(Z), Res)
		for i in Fx:
			for j in Fy:
				for k in Fz:
					d = np.sqrt((X-i)**2 + (Y-j)**2	+ (Z - k)**2)
					d2 = sorted(d)
					d3 = d2[0:K]
					d_k.append(d3)	
	else:	
		print 'No aveilable dimension'
    	print 'Completed neighbours finder'
    #print d4[0:10]
    
Neighbours_Cartesian(K, Res)


def solution(K):
	k = range(1, 100)
	global d_0
    	d_0 = []
#	T = []
	#T2 = []
	if D == 1:
		for i in range(len(d_k)):
			T = []
			for j in range(K):
				teo = d_k[i][j]/k[j]
				T.append(teo)
			T2 = sum(T)
			d_0.append(1.0/(T2*T2*np.pi))
	elif D == 2:
    		for i in range(len(d_k)): #escala los puntos del espacio
        		T = []
			for j in range (K):# j escala los vecinos
            			teo = d_k[i][j]**2 / k[j] 
            			T.append(teo)
				#print type(T)
        		T2 = np.sqrt(sum(T)) 
        		d_0.append(K/(T2*T2*np.pi)) #This is divided in order to get n_0
	else:
		print 'No aveilable dimension'
	print 'Completed density estimation'
	
solution(K)

#Sigma Estimator

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
    #p  rint type(sigma), dtype(S[1]), type(S), dtype(S2), type(sigma_2)
    print 'Completed Sigma estimation'
    print 'Writting data in (../output/)'
sigma_estimator(K)


# Writting Data to make plots

def plots(Res):
	X = data[:, N_X]/15.0
        Y = data[:, N_Y]
        if D ==3:
		Z = data[:, N_Z]

	f = open(str(data_output), 'w')
    	f.write("#x     y         d_0       sigma" + "\n")
    	#global x
    	#global y
    	x = []
    	y = []
    	Fx = np.linspace(np.amin(X) , np.amax(X)  , Res)
    	Fy = np.linspace(np.amin(Y) , np.amax(Y)  , Res)
    	#FRx = linspace(amin(R*cos(theta)) - 5, amax(R*cos(theta)) + 5 , Res)
    	#FRy = linspace(amin(R*sin(theta)) - 5, amax(R*sin(theta)) + 5 , Res)
    	for i in Fx:
        	for j in Fy:
            		x.append(i)
            		y.append(j)
    	for i in range(len(x)):
        	f.write(str(x[i]) + "  " + str(y[i]) + "  " + str(d_0[i]) + "  " + str(sigma_2[i])+"\n") 
    	f.close()
    #print len(x), len(y), len(T3)

plots(Res)


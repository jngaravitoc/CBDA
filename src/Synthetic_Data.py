import numpy as np
from math import *
from random import *

def Init_Distribution(Dx, Dy, nx, ny):
    f = open('../input/synthetic_data.dat', 'w')
    global X
    global Y
    X = np.random.random(nx)*Dx
    Y = np.random.random(ny)*Dy
    X = np.append(X, np.random.normal(loc= 0.80*Dx, scale = 2, size = 300))
    Y = np.append(Y, np.random.normal(loc= 0.60*Dy, scale = 8, size = 300))
    X = np.append(X, np.random.normal(loc= 0.20*Dx, scale = 5, size = 100))
    Y = np.append(Y, np.random.normal(loc= 0.30*Dy, scale = 5, size = 100))
    for i in range(nx):
        f.write(str(X[i]) + "  " + str(Y[i]) + "\n")
    f.close()
   #figure(num=None, figsize=(9.5,9))
    #xlabel('$\mathrm{x}$', fontsize = 35)
    #ylabel('$\mathrm{y}$', fontsize = 35)
    #title('$\mathrm{Random}$ $\mathrm{Cartesian}$ $\mathrm{Distribution}$ ', fontsize = 40)
    #tick_params(axis='both', which='major', labelsize=18)
    #scatter(X, Y)
    #ylim([0, ny])
    #xlim([0, nx]) 
    #savefig('RandomCartesianDistribution.png')
    #return X, Y
    
def Init_Distribution_Polar(nR):
    global R
    global theta
    R = np.random.random(1000)*nR
    theta = (np.random.random(1000)+1)*randrange(0, 360)
    #figure(num=None, figsize=(9.5,9))
    #xlabel('$\mathrm{x}$', fontsize = 35)
    #ylabel('$\mathrm{y}$', fontsize = 35)
    #title('$\mathrm{Random}$ $\mathrm{Polar}$ $\mathrm{Distribution}$', fontsize = 40)
    #tick_params(axis='both', which='major', labelsize=18)
    #scatter(R*cos(theta), R*sin(theta))
    #xlim([-120, 120])
    #ylim([-120, 120])
    #savefig('RandomPolarDistribution.png')
    return R, theta

#nit_Distribution(100, 100, 10000, 10000)
def Init_Distribution3D(Dx, Dy, Dz, nx, ny, nz):
    f = open('../input/3Dsynthetic_data.dat', 'w')
    global X
    global Y
    global Z
    X = np.random.random(nx)*Dx
    Y = np.random.random(ny)*Dy
    Z = np.random.random(nz)*Dz
    X = np.append(X, np.random.normal(loc= 0.80*Dx, scale = 2, size = 300))
    Y = np.append(Y, np.random.normal(loc= 0.60*Dy, scale = 8, size = 300))
    X = np.append(X, np.random.normal(loc= 0.20*Dx, scale = 5, size = 100))
    Y = np.append(Y, np.random.normal(loc= 0.30*Dy, scale = 5, size = 100))
    for i in range(nx):
        f.write(str(X[i]) + "  " + str(Y[i]) + "   " + str(Z[i]) + "\n")
    f.close()
 
Init_Distribution3D(100, 100, 100, 10, 10, 10)

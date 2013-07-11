from matplotlib import pyplot as plt
from math import *
import numpy as np

density = np.loadtxt('../output/DensityData.txt')
#data = np.loadtxt('../input/synthetic_data.dat')
#data = np.loadtxt('/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/quest.rrab.cmj_akv.dat')
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/layden.rrls.mateu_dists.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/sesar.rrls.mydists.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/JJD_BD.dat")
data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/JJD_VLMS.dat")


x = density[:,0]
y = density[:,1]
n_0 = density[:,2]
X = data[:,0]
Y = data[:,1]

#plt.figure(num=None, figsize=(9.5,9))

xmin = np.amin(x)
xmax = np.amax(x) 
ymin = np.amin(y) 
ymax = np.amax(y)

print 'xmin=', np.amin(x)
print 'xmax=', np.amax(x)
print 'ymin=', np.amin(y)
print 'ymax=', np.amax(y)

#Rxmin = np.amin(R*cos(theta))
#Rxmax = np.amax(R*cos(theta)) 
#Rymin = np.amin(R*sin(theta)) 
#Rymax = np.amax(R*sin(theta))

#plt.xlabel('$\mathrm{RA}$', fontsize = 35)
#plt.ylabel('$\mathrm{Dec}$', fontsize = 35)
#plt.title('$\mathrm{\sigma}$', fontsize = 38)
#plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
hist, xedges, yedges = np.histogram2d(x, y, bins=(200, 200),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
hist = hist.transpose()
my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
plt.imshow(np.log10(hist), extent=[xmin, xmax, ymin, ymax], interpolation = 'nearest', origin='lower')
plt.colorbar(shrink = 0.05)    #this shrink is to set the scale of the colorbar
plt.scatter(X , Y, s = 1.5)
plt.show()
#aspect = auto!!




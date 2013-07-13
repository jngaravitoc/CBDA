from matplotlib import pyplot as plt
from math import *
import numpy as np

density = np.loadtxt('../output/DensityData_Quest.txt')
density2 = np.loadtxt('../output/DensityData_Synthetic_QuestX10.txt')
#data = np.loadtxt('../input/synthetic_data.dat')
data = np.loadtxt('/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/quest.rrab.cmj_akv.dat')
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/layden.rrls.mateu_dists.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/sesar.rrls.mydists.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/JJD_BD.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/JJD_VLMS.dat")
data2 = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/synth.quest.x10.dat")


x = density[:,0]
y = density[:,1]
n_0 = density[:,2]
X = data[:,0]
Y = data[:,4]

x2 = density2[:,0]
y2 = density2[:,1]
n_02 = density2[:,2]
X2 = data2[:,0]
Y2 = data2[:,4]



fig = plt.figure(num=None, figsize=(19.5,9))

ax = fig.add_subplot(111)

xmin = np.amin(x)
xmax = np.amax(x) 
ymin = np.amin(y) 
ymax = np.amax(y)

xmin2 = np.amin(x2)
xmax2 = np.amax(x2)
ymin2 = np.amin(y2)
ymax2 = np.amax(y2)

print 'xmin=', np.amin(x)
print 'xmax=', np.amax(x)
print 'ymin=', np.amin(y)
print 'ymax=', np.amax(y)

print 'x2min=', np.amin(x2)
print 'x2max=', np.amax(x2)
print 'y2min=', np.amin(y2)
print 'y2max=', np.amax(y2)

#Rxmin = np.amin(R*cos(theta))
#Rxmax = np.amax(R*cos(theta)) 
#Rymin = np.amin(R*sin(theta)) 
#Rymax = np.amax(R*sin(theta))

#plt.xlabel('$\mathrm{RA}$', fontsize = 35)
#plt.ylabel('$\mathrm{Dec}$', fontsize = 35)
#plt.title('$\mathrm{\sigma}$', fontsize = 38)
#plt.tick_params(axis='both', which='major', labelsize=18)
#x.set_tick..
#ax.set_tick_params(axis='both', which='major', labelsize=18)
hist, xedges, yedges = np.histogram2d(x, y, bins=(200, 200),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
hist = hist.transpose()
my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
#ax.imshow((hist), extent=[xmin, xmax, ymin, ymax], interpolation = 'gaussian', origin='lower', aspect='auto')

hist2, xedges2, yedges2 = np.histogram2d(x2, y2, bins=(200, 200),range=[[xmin2,xmax2],[ymin2,ymax2]], weights = n_02)
hist2 = hist.transpose()
my_extent = (xedges2[0], xedges2[-1], yedges2[0], yedges2[-1])
ax.imshow((hist/hist2), extent=[xmin2, xmax2, ymin2, ymax2], interpolation = 'gaussian', origin='lower', aspect='auto')
#plt.colorbar(shrink = 0.05)    #this shrink is to set the scale of the colorbar
plt.xlim([60, 140])
plt.ylim([0, 37])
#scatter(X , Y, s = 1.5)
#plt.savefig("../output/plots/Synthetic_QuestX100_cut_NOLOG.png")
plt.show()





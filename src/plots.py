from matplotlib import pyplot as plt
from math import *
import numpy as np


# Defining data paths.

density = np.loadtxt('../output/Layden_ARvsRhel.txt')  # This is the data comming from CBDA.py and have the density and the sigma
density2 = np.loadtxt('../output/DensityData_SynthQuestX100_ARvsRhel.txt')
#ata = np.loadtxt('../input/synthetic_data.dat')  # this is the Data to make the scatter plot. (Initial distribution)
#data = np.loadtxt('/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/Quest_cut.dat')
data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/layden.rrls.mateu_dists.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/sesar.rrls.mydists.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/JJD_BD.dat")
#data = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/JJD_VLMS.dat")
data2 = np.loadtxt("/home/nicolas/Dropbox/github/Bayesian_Homeworks/CBDA/input/synth.quest.x100.dat")



global Out_fig_path
Out_fig_path  = '../output/plots/Layden_RAvsRhel.png'

def histogram(Out_fig_path, Res, X_label, Y_label, Title, Fig_xsize, Fig_ysize, Xdata_row, Ydata_row):

	x = density[:,0]
	y = density[:,1]
	n_0 = density[:,2]
	X = data[:, Xdata_row]/15  #Divide by 15 when expressing AR in hours
	Y = data[:, Ydata_row]

	fig = plt.figure(num=None, figsize=(Fig_xsize,Fig_ysize))
	ax = fig.add_subplot(111)
	xmin = np.amin(x)
	xmax = np.amax(x) 
	ymin = np.amin(y) 
	ymax = np.amax(y)

	print 'xmin=', np.amin(x)
	print 'xmax=', np.amax(x)
	print 'ymin=', np.amin(y)
	print 'ymax=', np.amax(y)

	plt.xlabel('$\mathrm{' + str(X_label) + '}$', fontsize = 35)
	plt.ylabel('$\mathrm{' + str(Y_label) + '}$', fontsize = 35)
	plt.title('$\mathrm{'+ str(Title) + '}$', fontsize = 38)
	plt.tick_params(axis='both', which='major', labelsize=18)
	#ax.set_tick_params(axis='both', which='major', labelsize=18)
	hist, xedges, yedges = np.histogram2d(x, y, bins=(Res, Res),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
	hist = hist.transpose()
	my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
	ax.imshow(np.log10(hist), extent=[xmin, xmax, ymin, ymax], interpolation = 'gaussian', origin='lower', aspect='auto')
	plt.scatter(X , Y, s = 1.5)
	#plt.xlim([20, 24])
	#plt.xlim([0, 4])
	plt.savefig(str(Out_fig_path))
	plt.show()


histogram(Out_fig_path, 200, 'AR', 'R_{hel}', 'Layden Catalogue Density', 9.5, 9.5,  0,  4)


  

def residuos(Out_fig_path, Res, X_label, Y_label, Title, Fig_xsize, Fig_ysize, Xdata_row, Ydata_row, Xdata2_row, Ydata2_row):
	x = density[:,0]
	y = density[:,1]
	n_0 = density[:,2]
	X = data[:, Xdata_row]
	Y = data[:, Ydata_row]
	
	x2 = density2[:, 0]
	y2 = density2[:, 1]
	n_02 = density[:,2]
	X2 = data2[:, Xdata2_row]
	Y2 = data2[:, Ydata2_row] 

	fig = plt.figure(num=None, figsize=(Fig_xsize, Fig_ysize))
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

	
	plt.xlabel('$\mathrm{' + str(X_label) + '}$', fontsize = 35)
	plt.ylabel('$\mathrm{' + str(Y_label) + '}$', fontsize = 35)
	plt.title('$\mathrm{'+ str(Title) + '}$', fontsize = 38)
	plt.tick_params(axis='both', which='major', labelsize=18)
	#ax.set_tick..
	#ax.set_tick_params(axis='both', which='major', labelsize=18)
	hist, xedges, yedges = np.histogram2d(x, y, bins=(Res, Res),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
	hist = hist.transpose()
	my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
	hist2, xedges2, yedges2 = np.histogram2d(x2, y2, bins=(200, 200),range=[[xmin2,xmax2],[ymin2,ymax2]], weights = n_02)
	hist2 = hist.transpose()
	my_extent = (xedges2[0], xedges2[-1], yedges2[0], yedges2[-1])
	ax.imshow((hist*10/hist2), extent=[xmin2, xmax2, ymin2, ymax2], interpolation = 'gaussian', origin='lower', aspect='auto')
	#colorbar(shrink = 0.05)    #this shrink is to set the scale of the colorbar
	plt.savefig(str(Out_fig_path))
	plt.show()


#residuos(Out_fig_path, 200, 'AR', 'R_{hel}', 'Residuos  Quest/Questx100', 9.5, 9.5,  0,  1, 0, 4)

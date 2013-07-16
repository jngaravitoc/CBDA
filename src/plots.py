from matplotlib import pyplot as plt
from math import *
import numpy as np
import sys
import scipy as sc

if len(sys.argv) < 2:
        print 'please provide a script with the following information:'
else:
        x = sc.genfromtxt(sys.argv[1], dtype='S')
	dic = {}
	for i in range(len(x[:,0])):
		(key, val) = (x[i,0], x[i, 2])
		dic[(key)] = val
	#ic = {}
        #ifile =  open(sys.argv[1], 'r') 
        
#	for line in ifile.readline():
#		print line
#		(key, val) = line.split(' = ')
 #               dic[(key)] = val
#print di
hist = float(dic['Histogram'])
Residues = float(dic['Residues'])
density_1 = np.loadtxt(dic['densities_path_1'])
#density_2 = np.loadtxt(dic['densities_path_2'])
data_1 = np.loadtxt(dic['data_path_1'])
#data_2 = np.loadtxt(dic['data_path_2'])
out_fig_path = dic['out_fig_path']
n_bins = float(dic['n_bins'])
histo_xlabel = dic['histo_xlabel']
histo_ylabel = dic['histo_ylabel']
histo_title = dic['histo_title']
histo_xsize = float(dic['histo_xsize'])
histo_ysize = float(dic['histo_ysize'])
x_data_row = float(dic['x_data_row'])
y_data_row = float(dic['y_data_row'])

def histogram(out_fig_path, n_bins, histo_xlabel, histo_ylabel,histo_title, histo_xsize, histo_ysize, x_data_row, y_data_row):
	x = density_1[:,0]
	y = density_1[:,1]
	n_0 = density_1[:,2]
	X = data_1[:,x_data_row]/15  #Divide by 15 when expressing AR in hours
	Y = data_1[:,y_data_row]

	fig = plt.figure(num=None, figsize=(histo_xsize,histo_ysize))
	ax = fig.add_subplot(111)
	xmin = np.amin(x)
	xmax = np.amax(x) 
	ymin = np.amin(y) 
	ymax = np.amax(y)

	print 'xmin=', np.amin(x)
	print 'xmax=', np.amax(x)
	print 'ymin=', np.amin(y)
	print 'ymax=', np.amax(y)

	plt.xlabel('$\mathrm{' + str(histo_xlabel) + '}$', fontsize = 35)
	plt.ylabel('$\mathrm{' + str(histo_ylabel) + '}$', fontsize = 35)
	plt.title('$\mathrm{'+ str(histo_title) + '}$', fontsize = 38)
	plt.tick_params(axis='both', which='major', labelsize=18)
	#ax.set_tick_params(axis='both', which='major', labelsize=18)
	hist, xedges, yedges = np.histogram2d(x, y, bins=(n_bins, n_bins),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
	hist = hist.transpose()
	my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
	ax.imshow((hist), extent=[xmin, xmax, ymin, ymax], interpolation = 'gaussian', origin='lower', aspect='auto')
	plt.scatter(X , Y, s = 1.5)
	#plt.xlim([20, 24])
	#plt.xlim([0, 4])
	plt.savefig(str(out_fig_path))
	plt.show()

if hist == 1:
	histogram(out_fig_path, n_bins, histo_xlabel, histo_ylabel,histo_title, histo_xsize, histo_ysize, x_data_row, y_data_row)

 

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
	# generalizar a mismos xmin y xmax
	ist, xedges, yedges = np.histogram2d(x, y, bins=(Res, Res),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
	hist = hist.transpose()
	my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
	hist2, xedges2, yedges2 = np.histogram2d(x2, y2, bins=(Res, Res),range=[[xmin2,xmax2],[ymin2,ymax2]], weights = n_02)
	hist2 = hist.transpose()
	my_extent = (xedges2[0], xedges2[-1], yedges2[0], yedges2[-1])
	im = ax.imshow((hist*10/hist2), extent=[xmin2, xmax2, ymin2, ymax2], interpolation = 'gaussian', origin='lower', aspect='auto')
	plt.colorbar(im, shrink = 0.05)    #this shrink is to set the scale of the colorbar
	plt.savefig(str(Out_fig_path))
	plt.show()

if Residues == 1:
	residuos(Out_fig_path, 200, 'AR', 'R_{hel}', 'Residuos  Quest/Questx100', 9.5, 9.5,  0,  4, 0, 4)

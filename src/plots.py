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

# Defining Variables
	hist = float(dic['Histogram'])
	Residues = float(dic['Residues'])
	density_1 = np.loadtxt(dic['densities_path_1'])
	data_1 = np.loadtxt(dic['data_path_1'])
	nx_bins = float(dic['nx_bins'])
	ny_bins = float(dic['ny_bins'])
	D = int(dic['D'])
	if hist == 1:
		out_histofig_path = dic['out_histofig_path']
		histo_xlabel = dic['histo_xlabel']
		histo_ylabel = dic['histo_ylabel']
		histo_title = dic['histo_title']
		histo_xsize = float(dic['histo_xsize'])
		histo_ysize = float(dic['histo_ysize'])
		x_data_row = float(dic['x_data_row'])
		y_data_row = float(dic['y_data_row'])

	if Residues == 1:
		density_2 = np.loadtxt(dic['densities_path_2'])
		data_2 = np.loadtxt(dic['data_path_2'])
		out_residuesfig_path = dic['out_residuesfig_path']
		res_xlabel = dic['res_xlabel']
		res_ylabel = dic['res_ylabel']
		res_title = dic['res_title']
		res_xsize = float(dic['res_xsize'])
		res_ysize = float(dic['res_ysize'])
		x_data_row = float(dic['x_data_row'])
        	y_data_row = float(dic['y_data_row'])
		x_data2_row = float(dic['x_data2_row'])
		y_data2_row = float(dic['y_data2_row'])

	if D == 1:
		out_scatter = dic['out_scatter']


def histogram(out_histofig_path, nx_bins, ny_bins, histo_xlabel, histo_ylabel,histo_title, histo_xsize, histo_ysize, x_data_row, y_data_row):
	x = density_1[:,0]
	y = density_1[:,1]
	n_0 = density_1[:,2]
	X = data_1[:,x_data_row]  #Divide by 15 when expressing AR in hours
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
	plt.title('$\mathrm{'+ str(histo_title).replace("_", "\ ") + '}$', fontsize = 38)
	plt.tick_params(axis='both', which='major', labelsize=18)
	#ax.set_tick_params(axis='both', which='major', labelsize=18)
	hist, xedges, yedges = np.histogram2d(x, y, bins=(nx_bins, ny_bins),range=[[xmin,xmax],[ymin,ymax]], weights = n_0)
	hist = hist.transpose()
	my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
	im = ax.imshow((hist), extent=[xmin, xmax, ymin, ymax], interpolation = 'gaussian', origin='lower', aspect='auto')
	plt.scatter(X , Y, s = 1.5)
	plt.ylim([0,35])
	plt.xlim([45, 160])
	plt.colorbar(im, shrink = 1.0)
	plt.savefig(str(out_histofig_path))
	plt.show()

if D == 2:
	print 'check'
	if hist == 1:
		histogram(out_histofig_path, nx_bins,ny_bins , histo_xlabel, histo_ylabel,histo_title, histo_xsize, histo_ysize, x_data_row, y_data_row)

 

def residuos(out_residuesfig_path, nx_bins, ny_bins, res_xlabel, res_ylabel, res_title, res_xsize, res_ysize, x_data_row, y_data_row, x_data2_row, y_data2_row):
	x = density_1[:,0]
	y = density_1[:,1]
	n_0 = density_1[:,2]
	X = data_1[:, x_data_row]
	Y = data_1[:, y_data_row]
	
	x2 = density_2[:, 0]
	y2 = density_2[:, 1]
	n_02 = density_2[:,2]
	X2 = data_2[:, x_data2_row]
	Y2 = data_2[:, y_data2_row] 

	fig = plt.figure(num=None, figsize=(res_xsize, res_ysize))
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

	
	plt.xlabel('$\mathrm{' + str(res_xlabel) + '}$', fontsize = 35)
	plt.ylabel('$\mathrm{' + str(res_ylabel) + '}$', fontsize = 35)
	plt.title('$\mathrm{'+ str(res_title).replace("_", "\ ") + '}$', fontsize = 38)
	plt.tick_params(axis='both', which='major', labelsize=18)
	N_0 = n_0 - n_02
	hist, xedges, yedges = np.histogram2d(x, y, bins=(nx_bins, ny_bins),range=[[xmin ,xmax],[ymin, ymax]], weights = N_0)
	hist = hist.transpose()
	my_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
	im = ax.imshow((hist), extent=[xmin, xmax, ymin, ymax], interpolation = 'gaussian', origin='lower', aspect='auto')
	plt.colorbar(im, shrink = 1.0)    
	plt.savefig(str(out_residuesfig_path))
	plt.show()
	
if D ==2:
	if Residues == 1:
		residuos(out_residuesfig_path, nx_bins, ny_bins, res_xlabel, res_ylabel, res_title, res_xsize, res_ysize,  x_data_row, y_data_row, x_data2_row, y_data2_row)


def one_d(out_scatter):
	X = density_1[:, 0] 
        Y = density_1[:, 1]
	fig = plt.figure(num=None, figsize=(9.5, 9))
        ax = fig.add_subplot(111)
	plt.plot(X, Y)
	plt.title('$\mathrm{Observational Data Quest 1D}$', fontsize = 35)
	plt.xlabel('$\mathrm{AR}$', fontsize = 35)
	plt.ylabel('$\mathrm{n_0}$', fontsize = 35)
	plt.savefig(str(out_scatter))
	plt.show()
if D ==1:	
	one_d(out_scatter)


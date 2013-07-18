import numpy as np 
from scipy import *
import matplotlib as plt
import math 
from random import *

data = np.loadtxt("../input/sesar.rrls.mydists.slims.par.ssurvey.sql1.dat")

X = data[:,4] 
Y = data[:,6] 

print 'Xmin = ', amin(X), 'Xmax = ',amax(X),'Ymin=', amin(Y), 'Ymax=' ,amax(Y)

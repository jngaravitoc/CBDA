import numpy as np 
from scipy import *
import matplotlib as plt
import math 
from random import *

data = np.loadtxt("../input/synth.quest.x1.dat")

X = data[:,0] 
Y = data[:,4] 

print 'Xmin = ', amin(X), 'Xmax = ',amax(X),'Ymin=', amin(Y), 'Ymax=' ,amax(Y)

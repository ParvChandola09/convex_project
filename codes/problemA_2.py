#Python libraries for math and graphics
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,savefig
import math
import matplotlib.pyplot as plt
from cvxpy import *

##########################################################
# Given N (Order of the filter) and wc (Cut off frequency),
# we need to minimize alpha (Attenuation constant) with
# alpha and Filter coefficients as objective varaibles 
##########################################################

N= 20 
sz = 15*N 
w = np.linspace(0,pi, sz)
wc = pi/2.5
wp = pi/3
k = np.arange(0,N+1).reshape(N+1,1)
coskw = cos(k*w).T
ih = np.zeros(sz) 

wi = np.where(w<= wp)[0].max()
wo = np.where(w>= wc)[0].min()

#ideal filter values
for i in range(0,wo):
    ih[i] = 1
    
# Create optimization variables
an = Variable((N+1,1))
alpha = Variable()

# Form objective function
obj = Minimize(alpha)

#Constraints
constraints = [ coskw[0:wi, :]@an <= 1.12, coskw[0:wi,:]@an >= 0.89, cos(wp*k).T@an >= 0.89, cos(wp*k).T@an <= 1.12,  coskw[wo:sz, :]@an <= alpha , coskw[wo:sz, :]@an >= -alpha, cos(wc*k).T@an <= alpha, cos(wc*k).T@an >= -alpha ]  

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve()

print(prob.status)

print("alpha=", alpha.value)
#print("an =", an.value)

taps = an.value[:,0]
#print(taps)

#finding the deviation from ideal response
hw = cos(k*w).T@taps
X = np.subtract(ih,hw)

#------------------------------------------------
# Plot the deviation of the filter.
#------------------------------------------------

figure(1)
clf()
plot(w, X, linewidth=2)
ylabel('Error factor')
xlabel('w in radians')
title('Deviation from ideal response for N=20 & wc=pi/2.5')
grid(True)
#savefig('./figs/dev1.png')
show()
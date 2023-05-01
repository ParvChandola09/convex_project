#Python libraries for math and graphics
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,savefig
import math
import matplotlib.pyplot as plt
from cvxpy import *

##############################################################
# Given wc (Cut off frequency) and alpha (Attenuation constant), 
# we need to minimize N (Order of the filter) with
# wc Filter coefficients as objective variables. We will start at minimum value of N 
##########################################################
alpha = 0.006

wc = pi/2.5
wp = pi/3


# Create optimization variables
N = 1
#Keep increasing N and check whether the solution exists or not.
while (True) :
    sz = N*15 
    w = np.linspace(0,pi, sz)
    k = np.arange(0,N+1).reshape(N+1,1)
    coskw = cos(k*w).T

    wi = np.where(w<= wp)[0].max()
    wo = np.where(w>= wc)[0].min()

    print("N=", N)
    an = Variable((N+1,1))
    # Form objective function
    obj = Minimize(0)
    
    #Constraints
    constraints = [ coskw[0:wi, :]@an <= 1.12, coskw[0:wi,:]@an >= 0.89, cos(wp*k).T@an >= 0.89, cos(wp*k).T@an <= 1.12,  coskw[wo:sz, :]@an <= alpha , coskw[wo:sz, :]@an >= -alpha, cos(wc*k).T@an <= alpha, cos(wc*k).T@an >= -alpha ]  

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()

    print(prob.status)
    if (prob.status == "optimal"):
        break
    N += 1 

taps = an.value[:,0]

#ideal filter
ih = np.zeros(sz) 
#ideal filter values
for i in range(0,wo):
    ih[i] = 1

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
title('Deviation from ideal response for w=pi/2.5 & alpha = 0.006')
grid(True)
#savefig('./figs/dev3.png')
show()
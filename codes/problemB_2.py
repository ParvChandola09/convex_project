#Python libraries for math and graphics
import numpy as np
from numpy import cos, sin, pi, absolute, arange
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show,savefig
import math
import matplotlib.pyplot as plt
from cvxpy import *
##############################################################
# Given N (Order of the filter and alpha (Attenuation constant), 
# we need to minimize wc (cut off frequency) with
# wc and Filter coefficients as objective variables 
##########################################################
N= 20
sz = N*15 
alpha = 0.006

w = np.linspace(0,pi, sz)
wc = pi/2.5
wp = pi/3

k = np.arange(0,N+1).reshape(N+1,1)
coskw = cos(k*w).T
ih = np.zeros(sz)

wi = np.where(w<= wp)[0].max()
wo = np.where(w>= wc)[0].min()



# Create optimization variables
an = Variable((N+1,1))
i = wi
omega = w[i]
#Keep increasing w and check whether the solution exists or not.
while (i <= sz) :
    # Form objective function
    hw = cos(k*omega).T@an
    obj = Minimize(hw)
    #Constraints
    constraints = [ coskw[0:wi, :]@an <= 1.12, coskw[0:wi,:]@an >= 0.89, cos(wp*k).T@an >= 0.89, cos(wp*k).T@an <= 1.12, coskw[i:, :]@an <= alpha, coskw[i:,:]@an >= -alpha,  hw <= alpha, hw >= -alpha ]  

    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()

    print(prob.status)
    if (prob.status == "optimal"):
        break
    i += 1 
    omega = w[i] 

taps = an.value[:,0]

print(omega)
#ideal filter values
womg = np.where(w>= omega)[0].min()
for i in range(0,womg):
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
title('Deviation from ideal response for N=20 & alpha = 0.006')
grid(True)
#savefig('./figs/dev2.png')
show()
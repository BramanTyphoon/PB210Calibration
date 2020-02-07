#.................Canonical example for pb210_calibration.py....................
# Author: James Bramante
# Date: September 10, 2019

import numpy as np
from matplotlib import pyplot as plt
import pb210_calibration as pbcal

# Use data from Appleby, P.G. (2001) "Chronostratigraphic techniques in
#   recent sediments." Tracking Environmental Change Using Lake Sediments

# Core depth in m
depth = np.array([0,0.5,2.5,4.5,6.5,8.5,10.5,12.5,14.5,16.5,18.5,\
        20.5,22.5,24.5,25.5,26.5, 27.5,28.5,32.5])/100
# Cumulative dry weight, kgm^-2
mass = np.array([0,0.03,0.21,0.42,0.7,1.04,1.38,1.67,1.98,\
        2.3,2.63,2.99,3.4,3.82,4.02,4.23,4.46,4.71,5.72])/1000*100**2
# Total 210Pb activity, Bqkg^-1
total = np.array([0,404, 430, 415,373,157,207,226,144,137,74,\
        40,26,13,11,17,6,6,-1])
# Error in total 210Pb activity, Bqkg^-1
error = np.array([0,31,24,24,25,20,28,25,18,14,14,6,9,6,9,6,6,6,5])
# Sediment age and rate of deposition per Appleby (2001)
t = np.array([0,1,4,9,17,25,32,41,53,66,82,97,114,131,138,150,167,181])
r = np.array([0,0.052,0.044,0.039,0.034,0.063,0.038,0.026,0.029,0.02,0.023,\
                0.026,0.024,0.028,0.028,0.012,0.021,0.013])/1000*100**2
# Depth below which total activity asymptotes to supported/equilibrium activity
bkgrd = 32.5/100

# Calculate density from Appleby (2001) mass data. pb210_calibration.py will
# calculate cumulative dry weight for you
density = np.diff(mass)/np.diff(depth)
for ii in np.arange(1,np.size(density)):
    density[ii] = density[ii]*2 - density[ii-1]
density = np.hstack((np.array([density[0]]),density))

# Instantiate a CRS model with the 210Pb data
pbwincrs = pbcal.pb210_calibration(total[1:],error[1:],depth[1:],density[1:],bkgrd,'crs')
pbwincrs.calibrate()
pbwincrs.plot([])

# Comapre our model output with the example data from Appleby (2001)
fig, axes = plt.subplots(2,1)
axes[0].plot(np.nanmedian(pbwincrs.data['age'][:-1,:],axis=1),t[1:])
axes[0].plot(np.array([0.,200.]),np.array([0.,200.]),'r--')
axes[0].set_xlabel('CRS Modeled age (yr before core recovery)')
axes[0].set_ylabel('Actual age (yr before core recovery)')
axes[1].plot(np.nanmedian(pbwincrs.data['rate'][:-1,:],axis=1),r[1:]/density[1:-1],'bs')
axes[1].plot(np.array([0.,0.01]),np.array([0.,0.01]),'r--')
axes[1].set_xlabel(r'CRS Modeled deposition rate (myr$^{-1}$)')
axes[1].set_ylabel(r'Actual deposition rate (myr$^{-1}$)')
plt.show()


# Do the same with a CIC model
pbwincic = pbcal.pb210_calibration(total[1:],error[1:],depth[1:],density[1:],bkgrd,'cic')
pbwincic.calibrate()
pbwincic.plot([])

# Compare our model output with the example data from Appleby (2001)
fig,axes = plt.subplots(1,1)
axes.plot(np.nanmedian(pbwincic.data['age'][:-1,:],axis=1),t[1:])
axes.plot(np.array([0.,200.]),np.array([0.,200.]),'r--')
axes.set_xlabel('CIC Modeled age (yr before core recovery)')
axes.set_ylabel('Actual age (yr before core recovery)')
plt.show()


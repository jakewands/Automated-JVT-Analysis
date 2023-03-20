# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:16:04 2023

@author: jwands
"""
#This python code automatically processes temperature-dependent JV data from solar cells to calculate the ideality factor (n), reverse
#saturation current density (J0), diode activation energy, diode activation energy temperature coefficienct (alpha), and reverse saturation
#current density prefactor (J00). The methods for calculating the values are outlined in:
#C. J. Hages,et al., “Generalized current-voltage analysis and efficiency limitations in non-ideal solar cells: Case of Cu 2 ZnSn(S x Se 1−x ) 4
#and Cu 2 Zn(Sn y Ge 1−y )(S x Se 1−x ) 4,” J. Appl. Phys., vol. 115, no. 23, p. 234504, Jun. 2014, doi: 10.1063/1.4882119.
#and:
#J. Wands, et al., “Stability of Cu(InxGa1-x)Se2 Solar Cells Utilizing RbF Post-Deposition Treatment Under a Sulfur Atmosphere,”
#Advanced Energy and Sustainability Research, Submitted.

#As the code is written it assumes there is an individual .txt IV file for each temperature held in a folder with no other .txt files.
#The files should have two columns: voltage and current. For automated extraction of temperature from the file name, end your file names
#with the measurement temperature to two decimal places multiplied by 100 (for T=298.15 end the file with 29815.txt). Otherwise enter
#temperature values by hand in the variable "darkT" and make sure they are in the same order as the program loads the files into the 
#variable "files". The program assumes the data is IV not JV. Set the cell area in the variable "area". If the files are JV change
#"area" to 1.

#Import all of the necessary packages for this analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd
import os
import glob

#Define initial variables. My raw data is in amps and needs to be converted to A/m^2. Enter your cell area or set to 1 if your data is already current density
area = 0.5e-4
k = 8.62e-5
n = []
j0 = []
r2 = []
s = []
win = []

#Define a linear and exponential function for fitting commands
def linfunc(x,a,b):
    return a*x+b
def expfunc(x,a,b):
    return a*np.exp(b*x)

#Set the folder with the .txt files
os.chdir(r'Example file path')
#Gather all files with .txt extension and put the file extension in a variable called "files"
files = glob.glob('*.txt')
#Automatically extract the temperature from file names. Alternatively you can enter your own temps in darkT and comment out the for loop
darkT = []
for i in range(len(files)):
    darkT.append(int(files[i][-9:-4])/100)
#Load the first file into darkdata and extract the voltage column into darkrawdata
darkdata = pd.read_csv(files[0], sep='\t')
darkrawdata = darkdata['V']
#Loop through "files" and extract the current column and add it to darkrawdata. The result is a dataframe with volts followed by the current for each temp
for i in range(len(files)):
    data2 = pd.read_csv(files[i], sep='\t')
    data2 = data2.divide(other=area)
    darkrawdata = pd.concat([darkrawdata,data2['I']],axis=1)

#This for loop calculates the dV/dJ curves, finds the linear region of a semi-log plot, fits a linear function, calculates a JV curve from the fit region, and extracts n and J0 from that curve
for z in range(len(files)):
    #Calculate dV/dJ
    dvdj = [(darkrawdata['V'][i+1]-darkrawdata['V'][i])/(darkrawdata.iloc[i+1,z+1]-darkrawdata.iloc[i,z+1]) for i in range(len(darkrawdata['V'])-1)]
    #Find the V=0 point
    zerovolt = darkrawdata.index[darkrawdata['V'] == 0].tolist()
    #Calculate the mean dV/dJ for +- 7 data points from V=0. This is used as shunt resistance and subtracted from dV/dJ.
    dvdj = [1/((1/dvdj[i])-(1/np.mean(dvdj[zerovolt[0]-7:zerovolt[0]+7]))) for i in range(len(dvdj))]
    logdvdj = [np.log(i) for i in dvdj]
    
    #The following loop checks the curve for linearity across a given window of datapoints. It loops from minwindow to maxwindow and prioritizes strong fits with larger windows. The window values can be changed to suit the data.
    minwindow = 5
    maxwindow = 50
    #Resets initial values for calculated parameters
    rsquared = 0                                                                                  
    slope = 0
    intercept = 0
    window = minwindow
    #First loop goes through range of windows
    for i in range(minwindow,maxwindow,1):
        #Second loop checks a given window at every datapoint in the dV/dJ curve
        for t in range(len(logdvdj)-i):
            #Linear regression performed on the current data range
            res = linregress(darkrawdata['V'][t:t+i], logdvdj[t:t+i])
            #Checks if the r^2 value is above 0.9 and the slope is sufficiently negative. Without the negative slope check it may accidently fit linear regions that aren't of interest. These values can be changed.
            if res.rvalue**2 > 0.9 and res.slope < -15:
                #Prioritizes larger windows and larger r^2 values to get the best fit over the most datapoints
                if i > window or res.rvalue**2 > rsquared:
                    rsquared = res.rvalue**2
                    slope = res.slope
                    intercept = res.intercept
                    window = i
                    windowstart = t
                    windowstop = t+i
    #Saves the r^2, slope, and window values for the final fit
    r2.append(rsquared)
    s.append(slope)
    win.append(window)
    
    #Calculates the dV/dJ curve for the main diode region given the previous fit
    dvdjmain = [expfunc(darkrawdata['V'][i],np.exp(intercept),slope) for i in range(len(darkrawdata['V']))]
    #Divides dV by dV/dJ to get a dJ curve
    djmain = [(darkrawdata['V'][i+1]-darkrawdata['V'][i])/dvdjmain[i] for i in range(len(dvdjmain)-1)]
    #Initializes the main diode JV curve to J=0 at V=0
    jmain = [0]
    #Adds dJ terms to create a J curve
    for i in range(len(darkrawdata['V'])-zerovolt[0]-1):
        jmain.append(jmain[i]+djmain[i+zerovolt[0]])
    #Fits exponential function to JV curve. The fit starts 50 datapoints in to reach the linear region and includes the next 100 datapoints. These values can be changed.
    popt, pcov = curve_fit(expfunc,darkrawdata['V'][zerovolt[0]+50:zerovolt[0]+150],jmain[50:150])
    #Calculates n and J0
    n.append(1/(k*darkT[z]*popt[1]))
    j0.append(popt[0])
    
    #Plot the dV/dJ curve along with the linear fit. Also plots the JV curve and fit. The ylim may need to be changed
    plt.figure(1)
    plt.subplot(121)
    plt.plot(darkrawdata['V'][:-1],logdvdj)
    plt.plot(darkrawdata['V'],linfunc(darkrawdata['V'],slope,intercept))
    plt.ylim((-11,3))    
    plt.subplot(122)
    plt.plot(darkrawdata['V'][zerovolt[0]:],jmain, 'r')
    plt.plot(darkrawdata['V'][zerovolt[0]:],expfunc(darkrawdata['V'][zerovolt[0]:],popt[0],popt[1]), 'k--')
    plt.yscale('log')
    plt.show()

#Calculates several values for the diode activation energy calculations
invkt = [1/(k*i) for i in darkT]
invnkt = [invkt[i]/n[i] for i in range(len(n))]
logj0 = [np.log(i) for i in j0]
logj0T2 = [np.log(j0[i]/darkT[i]**2) for i in range(len(j0))]
logj0T2half = [np.log(j0[i]/darkT[i]**2.5) for i in range(len(j0))]
nlogj0 = [n[i]*logj0[i] for i in range(len(n))]
firstfit = linregress(invkt,nlogj0)

#Initialize parameters for iterative fitting process
j00 = 0
alpha = 0
alphaguess = 0.001
#While loop calculates activation energy fits between two plots until they converge. I've rounded the alpha value to 5 decimal places but this can be changed
while round(alpha, 5) != round(alphaguess,5):
    alphaguess = alpha
    fit1y = [logj0[i]-alphaguess/(n[i]*k) for i in range(len(n))]
    fit1 = linregress(invnkt,fit1y)
    j00 = np.exp(fit1.intercept)
    ea1 = abs(fit1.slope)
    fit2y = [n[i]*np.log(j0[i]/j00) for i in range(len(j0))]
    fit2 = linregress(invkt,fit2y)
    alpha = k*fit2.intercept
    ea2 = abs(fit2.slope)
#Calculates the alpha corrected J0 plot and fits a line to find the final activation energy value
j0alphacorrect = [np.log(j0[i]/j00)-alpha/(n[i]*k) for i in range(len(n))]
finalfit = linregress(invnkt,j0alphacorrect)
ea = abs(finalfit.slope)

#Plot of n vs T, log(J0) vs 1/kT, log(J0/T^2) vs 1/kT, log(J0/T^2.5) vs 1/kT
plt.figure(2)
plt.subplot(221)
plt.plot(darkT,n,'ko')
plt.subplot(222)
plt.plot(invkt,logj0,'ko')
plt.subplot(223)
plt.plot(invkt,logj0T2,'ko')
plt.subplot(224)
plt.plot(invkt,logj0T2half,'ko')
plt.show()

#Plot of n*log(J0) vs 1/kT, log(J0) vs 1/nkT, plot of the iterative process to find alpha/J00, log(J0/J00)-alpha/nk vs 1/nkT. Activation energy values are added to the plots
plt.figure(3)
plt.subplot(221)
plt.plot(invkt,nlogj0,'ko')
plt.plot(invkt,[linfunc(invkt[i],firstfit.slope,firstfit.intercept) for i in range(len(invkt))], 'r')
plt.annotate('$E_A= $'+str(round(abs(firstfit.slope,2))),(25,25),xycoords='axes pixels')
plt.subplot(222)
plt.plot(invnkt,logj0,'ko')
plt.subplot(223)
plt.plot(invkt,fit2y,'ko')
plt.plot(invkt,[linfunc(invkt[i],fit2.slope,fit2.intercept) for i in range(len(invkt))], 'r')
plt.annotate('$E_A= $'+str(round(abs(fit2.slope,2))),(25,25),xycoords='axes pixels')
plt.subplot(224)
plt.plot(invnkt,j0alphacorrect,'ko')
plt.plot(invnkt,[linfunc(invnkt[i],finalfit.slope,finalfit.intercept) for i in range(len(invnkt))], 'r')
plt.annotate('$E_A= $'+str(round(abs(finalfit.slope,2))),(25,25),xycoords='axes pixels')
plt.show()

#! /usr/bin/env python

##############################################################################
# ANICETO B. MAGHIRANG III                                                   #
# D092030009                                                                 #
#                                                                            #
# version 1.0                                                                #
# Thermoelectric Properties                                                  #
# Data processing (using 1 file, default: interpolation.trace) and plotting  #
##############################################################################

from __future__ import print_function
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pathlib import Path
import sys
# pd.set_option('display.max_rows', None)
pd.reset_option('display.float_format', silent=True)

######################## FUNCTIONS ##################################
E_proc = [] ### processed E (µ-µ_f)
kTot = [] ### kappa_e + kappa_L
def energy_kTot(ztdata, T_icp_kappaL_data): ### µ-µ_f & total kappa
    for data1 in ztdata:
        for data2 in T_icp_kappaL_data:
            if data1[1] == data2[0]:
                energy = data1[0] - data2[1]
                kappaTot = data1[7] + data2[2]
        E_proc.append(energy)
        kTot.append(kappaTot)

ZT = [] ### final DataFrame
def temp_subdivide(unique_temp): ### to subdivide all unique temperatures
    for temp in unique_temp:
        ZT.append(TE[TE['T'] == temp])
        
def exponent_formatter(min=-2, max=2): ### to improve tick aesthetics
    mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((min,max))
    plt.gca().yaxis.set_major_formatter(mf)        
    
def final_figures(legend): ### to plot the final 6 figures
    fig, axes = plt.subplots(nrows=2, ncols=3) #, figsize=(15, 10)
    fig.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.3)

    # (a) Seebeck Coefficient
    plt.subplot(2, 3, 1)
    plt.plot(ZT[2]['E_proc'], ZT[2]['S'], 'r-')
    plt.plot(ZT[4]['E_proc'], ZT[4]['S'], 'g-')
    plt.plot(ZT[6]['E_proc'], ZT[6]['S'], 'b-')
    plt.legend(legend, frameon=False)
    plt.xlim([x_min, x_max])
    #plt.ylim(-0.0006,0.0006)
    plt.xlabel('$µ-µ_{f}$ (eV)') ; plt.ylabel('S (x$10^{-3}$) V/K')
    exponent_formatter()

    # (b) Electrical Conductivity
    plt.subplot(2, 3, 2)
    plt.plot(ZT[2]['E_proc'], ZT[2]['sig'], 'r-')
    plt.plot(ZT[4]['E_proc'], ZT[4]['sig'], 'g-')
    plt.plot(ZT[6]['E_proc'], ZT[6]['sig'], 'b-')
    plt.legend(legend, frameon=False)
    plt.xlim([x_min, x_max]) ; plt.gca().set_ylim(bottom=0)
    #plt.ylim(0,300000)
    plt.xlabel('$µ-µ_{f}$ (eV)') ; plt.ylabel('σ (x$10^{5}$) 1/Ωm')
    exponent_formatter()

    # (c) Power Factor
    plt.subplot(2, 3, 3)
    plt.plot(ZT[2]['E_proc'], ZT[2]['PF'], 'r-')
    plt.plot(ZT[4]['E_proc'], ZT[4]['PF'], 'g-')
    plt.plot(ZT[6]['E_proc'], ZT[6]['PF'], 'b-')
    plt.legend(legend, frameon=False)
    plt.xlim([x_min, x_max]) ; plt.gca().set_ylim(bottom=0)
    #plt.ylim(0,0.002)
    plt.xlabel('$µ-µ_{f}$ (eV)') ; plt.ylabel('$S^{2}$σ (x$10^{-3}$) W/$mK^{2}$')
    exponent_formatter()

    # (d) Electronic Thermal Conductivity
    plt.subplot(2, 3, 4)
    plt.plot(ZT[2]['E_proc'], ZT[2]['kappaE'], 'r-')
    plt.plot(ZT[4]['E_proc'], ZT[4]['kappaE'], 'g-')
    plt.plot(ZT[6]['E_proc'], ZT[6]['kappaE'], 'b-')
    plt.legend(legend, frameon=False)
    plt.xlim([x_min, x_max]) ; plt.gca().set_ylim(bottom=0)
    #plt.ylim(0,3)
    plt.xlabel('$µ-µ_{f}$ (eV)') ; plt.ylabel('$κ_{e}$ W/mK')
    exponent_formatter()

    try:
        # (e) Lattice Thermal Conductivity
        plt.subplot(2, 3, 5)
        plt.plot(unique_temp[:], kappaL[:], 'ro-')
        plt.xlim(0,1300) ; #plt.ylim(0,0.3)
        plt.xlabel('$Temperature$ (K)') ; plt.ylabel('$κ_{l}$ W/mK')
        exponent_formatter()

        # (f) ZT Number
        plt.subplot(2, 3, 6)
        plt.plot(ZT[2]['E_proc'], ZT[2]['ZT'], 'r-')
        plt.plot(ZT[4]['E_proc'], ZT[4]['ZT'], 'g-')
        plt.plot(ZT[6]['E_proc'], ZT[6]['ZT'], 'b-')
        plt.legend(legend, frameon=False)
        plt.xlim([x_min, x_max]) ; plt.gca().set_ylim(bottom=0)
        #plt.ylim(0,3)
        plt.xlabel('$µ-µ_{f}$ (eV)') ; plt.ylabel('ZT Number')
    except:
        pass

#     plt.savefig("TE_plots.png", dpi=1200)
    plt.show()

################# FIGURE SPECS############################
plt.rcParams['figure.figsize'] = [20, 10] # Figure size
plt.rcParams['axes.linewidth'] = 2 # box width
plt.rcParams['lines.linewidth'] = 2 # line width

plt.rc('axes', labelsize=24) # label font size
plt.rc('xtick', labelsize=18) # xtick size
plt.rc('ytick', labelsize=18) # ytick size
plt.rc('legend',fontsize=18) # Legend font size

# legend
legend = ['300K', '500K', '700K']

######### CONSTANTS #########
ryd2eV = 13.6056980659      #
hart2eV = 27.211396641308   #
tau = 1E-14                 #
#############################

################# READ interpolation.trace ########################
# interpolation.trace (1 file processing)

interpolation_path = 'interpolation.trace'
interpolation = Path(interpolation_path)

if interpolation.is_file():
    zt = pd.read_csv(interpolation, sep='\s+', header=0)
    zt.columns = ['E','T','N','D','S','sig','R','kappaE','c','chi','']
else:
    print("interpolation.trace does not exist!")
    sys.exit()

################# DATA PROCESSING #################################

##### CONVERSION #####
# Rydberg to eV      #
zt['E'] *= ryd2eV    #
# remove tau         #
zt['sig'] *= tau     #
zt['kappaE'] *= tau  #
######################

# unique TEMP in interpolation.trace
unique_temp = pd.unique(zt['T'])

# seebeck^2 & PF
zt['S2'] = zt['S'] ** 2 ; S2 = zt['S2'].to_numpy()
zt['PF'] = zt['S2'] * zt['sig'] ; PF = zt['PF'].to_numpy()

# input intrinsic chemical potentials & kappaL if already finished
icp = [float(icp)*hart2eV for icp in input("Enter all intrinsic chemical potentials from Boltztrap : ").split()]
kappaL = [float(kappaL) for kappaL in input("Enter kappaL (if available) : ").split()] ### float(kappaL)*tau

T_icp_kappaL = pd.DataFrame(np.vstack([unique_temp, icp, kappaL]).T)
T_icp_kappaL.columns = ['T', 'icp', 'kappaL']

# µ-µ_f & total kappa
ztdata = zt.to_numpy()
T_icp_kappaL_data =  T_icp_kappaL.to_numpy()
energy_kTot(ztdata, T_icp_kappaL_data) # energy_kTot
E_proc = np.array(E_proc)
kTot = np.array(kTot)

# Processed interpolation.trace
ztdata = pd.DataFrame(ztdata)
E_proc = pd.DataFrame(E_proc)
kTot = pd.DataFrame(kTot)

# Final DataFrame
TE = pd.concat([ztdata, E_proc, kTot], axis=1, join='inner')
TE.columns = ['E','T','N','D','S','sig','R','kappaE','c','chi','', 'S2', 'PF', 'E_proc', 'kTot']
TE['ZT'] = (TE['T']*TE['PF'])/TE['kTot']

# subdivide the unique temperature data
temp_subdivide(unique_temp)


# xlim user input
x_min, x_max = [float(x) for x in input("Enter µ-µ_f (eV) range [e.g. -1.50 1.50]: ").split()] # input range
# x_min = -1.50 ; x_max = 1.50 # if desired range is already known

final_figures(legend)
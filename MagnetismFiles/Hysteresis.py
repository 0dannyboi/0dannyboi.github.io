'''
This program was used to analyze data collected during a lab studying magnetic hysteresis in ferromagnetic torroids.
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy
from shapely.geometry import Polygon # compute area for work 
from uncertainties import ufloat #handles propogation of errors
from numpy import pi
from uncertainties.umath import *


''' Experiment Parameters '''
mu_0 = 1.256637062 * 10 ** -6 #vacuum permeability
R2 = 5000 # resistance (in Ohms) of shunt resistor
omega = 2 * pi * 1000 #angular frequency of modulating AC signal

''' Uncertainty in Measured Values '''
sigma_dim = 0.05 * 10 ** -3 #uncertainty in length measurements of the toroids
sigma_R2 = 0.025 * R2 #resistor uncertainty value was provided as a percent
sigma_v_ac = 0.004 * 5.00 #uncertainty in voltage supply of function generator
# suppling a voltage with an ampliude of 5.00
sigma_omega = 0.001 * omega # uncertainty in supplied AC angular frequency = 0.1%
sigma_dIdt = np.sqrt((omega * sigma_v_ac / R2)**2 + (5 * sigma_omega /R2)**2 +\
                     (omega * 5 * sigma_R2 / (R2 ** 2))**2)# uncertainy in current modulation



class data:
    def __init__(self, name, number, d2, d1, h, DC, AC, pickup, file):
        d2 = d2 * 10 ** -3
        d1 = d1 * 10 ** -3
        h = h * 10 ** -3
        f = open(file, "r").read()
        array_txt = np.loadtxt(file, skiprows=1)
        file_t = np.transpose(array_txt)
        self.t = file_t[0]
        self.freq = file_t[1]
        self.omega = self.freq[0] * 2 * pi
        self.Vrms = file_t[2][0]
        self.Vo = self.Vrms * np.sqrt(2)
        self.Idc = (file_t[3])
        self.X = file_t[4]
        self.Y =  file_t[5]
        self.R = file_t[6]
        r2 = 0.5 * d2
        r1= 0.5 * d1
        r_avg = 0.5 * r2 + 0.5 * r1
        self.v = h * pi * (r2 ** 2 - r1 ** 2)
        self.Lo = pickup * mu_0 * AC * h * np.log(r2 / r1) / (2 * pi)
        self.dIdt = (self.Vo / R2) * self.omega * np.cos(self.omega * self.t)
        self.mu_prime =   self.Y / (self.Lo * self.dIdt)
        self.H_avg = (DC * self.Idc * np.log(r2 / r1)) / (2 * pi * (r2 - r1))
        self.H = DC * self.Idc / (2 * pi * r_avg)
        self.B = mu_0 * scipy.integrate.cumtrapz(self.mu_prime, self.H_avg,\
                                                 initial=0)
        self.d1 = np.transpose([list(self.H[1::]), list(self.B[1::])])
        self.region = Polygon(self.d1)
        self.are = self.region.area
        self.w = self.are * self.v
        self.name = name
        self.number = str(number)
        self.r_avg_1 = 0.5 * r2 + 0.5 * r1
        self.height = h
        self.r2 = r2
        self.r1 = r1
        self.sigma_Idc = 0.000035 * self.Idc
        self.sigma_H = self.sigma_Idc * DC / (2 * pi * r_avg)
        self.sigma_Lo = ((sigma_dim * AC * pickup * mu_0) / (2 * pi * r1 * r2)) \
                        * np.sqrt((h ** 2) * (r1 **2 + r2 ** 2) + (r1 * r2 *\
                                                       np.log (r2 / r1)) ** 2)
        self.sigma_Y = 0.01 * self.Y
        self.sigma_mu = (np.sqrt((sigma_dIdt * self.Lo * self.Y)**2 +\
                        (self.dIdt**2)*(self.sigma_Y**2 * self.Lo**2 +\
                 self.sigma_Lo**2 * self.Y**2))) / (self.dIdt**2 * self.Lo**2)
        
    
    def Hysteresis_loop(self, *args, **kwargs):
        plt.figure()
        plt.plot(self.H,  self.B)
        plt.xlabel("H [A / m]")
        plt.ylabel("B [T]")
        plt.grid()
        my_title = "Hysteresis loop for coil #" + self.number + ", " + self.name
        if ("title" in kwargs.keys()):
            my_title = kwargs["title"]
        plt.title(my_title)
        if ('hide' not in args):
            plt.show()
        if ('save' in args):
            plt.savefig(my_title + ".pdf")
    
    
    def run_calculations(self):
        sigma_dBi = [mu_0 * 0.5 * np.sqrt((self.H_avg[i] - self.H_avg[i - 1])**2 * \
                     (self.sigma_mu[i] + self.sigma_mu[i - 1])**2 + (self.mu_prime[i] + \
                      self.mu_prime[i - 1]) ** 2 * (self.sigma_H[i] + self.sigma_H[i - 1])**2) \
                     for i in range(0, len(self.H_avg))]
        sdBis = list(np.array(sigma_dBi) ** 2)
        sigma_B = [np.sqrt(sum(sdBis[0:i])) for i in range(0, len(self.sigma_mu))]
        negative_indices = np.where(self.B > 0)[0][:-1]
        region = [ [self.H_avg[i], self.B[i]] for i in negative_indices]
        A = 2 * Polygon(region).area
        sigma_dAi = 0.5 * np.array([np.sqrt((self.B[i+1] * self.sigma_H[i])**2 +\
                   (self.H_avg[i] * sigma_B[i+1])**2 + (self.B[i] * self.sigma_H[i+1])**2 +\
                   (self.H_avg[i+1] * sigma_B[i])**2) for i in negative_indices])
        sdAis = sigma_dAi ** 2
        sigma_A = np.sqrt(sdAis.sum())
        h = ufloat(self.height, sigma_dim)
        r2 = ufloat(self.r2, sigma_dim)
        r1 = ufloat(self.r1, sigma_dim)
        v = h * pi * (r2**2 - r1**2)
        W = ufloat(A, sigma_A) * v
        sigma_W = W.std_dev
        print("omega = " + str(self.omega))
        print("Vo = " + str(self.Vo))
        print("Area = " + str(A))
        print("Area Uncertainty = " + str(sigma_A))
        print("Volume = " + str(v.nominal_value))
        print("Volume Uncertainty = " + str(v.std_dev))
        print("Work = "+ str(W.nominal_value))
        print("Work Uncertainty = "+ str(W.std_dev))
    
    
    def quadrature(self, *args, **kwargs):
        plt.figure()
        plt.plot(self.Idc, 1 * self.Y)
        plt.xlabel("$I_{DC} ~ [A]$")
        plt.ylabel("$Y ~ [V]$")
        plt.grid()
        my_title = "Lock-in amp quadrature vs DC current for coil #" + \
            self.number + ", " + self.name
        if "title" in kwargs.keys():
            my_title = kwargs["title"]
        plt.title(my_title)
        if "hide" not in args:
            plt.show()
        if "save" in args:
            plt.savefig(my_title + ".pdf")
        
        
    def h_mu_prime(self, *args, **kwargs):
        plt.figure()
        plt.plot(self.H, self.mu_prime)
        plt.xlabel("H [A / m]")
        plt.ylabel("$\mu'$ [dimensionless]")
        plt.grid()
        my_title = "$\mu'$ vs H for coil #" + self.number + ", " + self.name
        if "title" in kwargs.keys():
            my_title = kwargs["title"]
        plt.title(my_title)
        if "hide" not in args:
            plt.show()
        if "save" in args:
            plt.savefig(my_title + ".pdf")
    
    
    def power_supply(self, *args, **kwargs):
        plt.figure()
        plt.plot(self.t, self.Idc)
        plt.xlabel("$t ~ [s]$")
        plt.ylabel("$I_{DC} ~ [A]$")
        plt.grid()
        my_title = "Power supply profile curve for coil #" + self.number + ", " + self.name
        if "title" in kwargs.keys():
            my_title = kwargs["title"]
        plt.title(my_title)
        if "hide" not in args:
            plt.show()
        if "save" in args:
            plt.savefig(my_title + ".pdf")

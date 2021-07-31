# NUEN 618 HW_2: Neutronics, hydraulics, and heat conduction with CN scheme source
import numpy as np
import copy
from scipy.optimize import fsolve
from properties import k_fuel
from properties import rhocp_fuel
from properties import cp_mod
from properties import rho_mod
from properties import rhocp_mod

class OS:
    def __init__(self, tstep):
        self.Ttrans  = 0.05
        self.tstrt   = 0       # sec
        self.tend    = 15*self.Ttrans      # sec
        self.tstep   = tstep#(self.tend - self.tstrt)/(self.numpnts - 1)
        self.numpnts = int((self.tend - self.tstrt)/self.tstep+1)#numpnts # number of points including 0 and 15
        self.time    = np.linspace(self.tstrt, self.tend, self.numpnts)# list(?) of discrete times

        self.l       = 0.1     # sec^{-1}
        self.b       = 600e-5
        self.L       = 1e-5    # sec^{-1}
        self.af      = -3.5e-5 # K^{-1}    alpha fuel
        self.ac      = -55e-5  # K^{-1}    alpha coolant
        self.rf      = 4.1e-3  # m         fuel radius
        self.rgap    = 4.11e-3 # m         fuel + gap radius
        self.rclad   = 4.75e-3 # m         fuel + gap + clad radius
        self.H       = 4       # m         fuel height
        self.pitch   = 1.26e-2 # m
        self.hgap    = 1e4     # W/(m^2 K) gap conductance
        self.kclad   = 17      # W/(m K)   clad conductivity
       #fuel conductivity
       #fuel density
       #fuel specific heat
        self.v       = 5       # m/s       coolant axial velocity
       #convective heat transfer coefficient
        self.kc      = 0.54    # W/(m K)   coolant conductivity
        self.mu      = 90e-6   # Pa s
       #coolant density
       #coolant specific heat
        self.Tc_in   = 290     # K         coolant fluid temperature

        self.p0      = 59506   # W
        self.p       = []
        self.c0      = self.b * self.p0 / self.l / self.L
        self.c       = []
        self.Tf0     = 0
        self.Tf      = []
        self.Tc0     = 0
        self.Tc      = []

        self.om      = 1 / self.H / np.pi / self.rf**2
        self.Af      = np.pi * self.rf**2# fuel area
        self.Aflow   = self.pitch**2 - np.pi * self.rclad**2
        self.Pwet    = 2 * np.pi * self.rclad
        self.Dhy     = 4 * self.Aflow / self.Pwet

        self.rho_ext = np.zeros(self.numpnts)
        for i in range(self.numpnts):
            t = self.time[i]
            if   t >= self.tstrt and t <= self.Ttrans:
                self.rho_ext[i] = 0
            elif t > self.Ttrans and t <= 2*self.Ttrans:
                self.rho_ext[i] = 1.45*self.b*(self.time[i]/self.Ttrans-1)
            else:
                self.rho_ext[i] = 1.45*self.b

    def convectiveHeat(self, Tc):
        '''Takes a single value for Tc'''
        dc, dcp   = rho_mod(Tc)
        Cpc, Cpcp = cp_mod(Tc)
        Re  = dc * self.v * self.Dhy / self.mu
        Pr  = Cpc * self.mu / self.kc
        hconv = 0.023 * Re**0.8 * Pr**0.4 * self.kc / self.Dhy
        return hconv

    def thermalResistance(self, Tf, Tc):
        hconv = self.convectiveHeat(Tc)
        kf, kfp = k_fuel(Tf)
        Rth = self.Af / 2 / np.pi * (1 / self.rgap / self.hgap +
              1 / self.kclad * np.log(self.rclad / self.rgap) +
              1 / self.rclad / hconv + 1 / 2 / kf)
        return Rth

    def reactivity(self, Tf, Tc, i):
        '''Called within reactorKinetics(), i is the time INDEX'''
        rho = self.rho_ext[i] + self.af * (Tf - self.Tf0) + \
                                self.ac * (Tc - self.Tc0)
        return rho

    def reactorKinetics(self, val, data, i):
        Tf = data[2]# scalar
        Tc = data[3]# scalar
        # i  = data[4]# scalar
        rho  = self.reactivity(Tf, Tc, i)
        D = (rho - self.b) / self.L
        p_ex = D * val[0] + self.l * val[1]
        c_ex = self.b / self.L * val[0] - self.l * val[1]
        xprsn = np.asarray((p_ex, c_ex))
        return xprsn

    def fuelTemperature(self, val, data, i):
        p = data[0]# scalar
        Tc = data[3]# scalar
        dfCpf, blah  = rhocp_fuel(val)
        Rth  = self.thermalResistance(val, Tc)
        xprsn = 1 / dfCpf * (self.om * p - (val - Tc) / Rth)
        return xprsn

    def coolantTemperature(self, val, data, i):
        Tf = data[2]# scalar
        dcCpf, blah   = rhocp_mod(val)
        Rth  = self.thermalResistance(Tf, val)
        xprsn = 1 / dcCpf / self.Aflow * (self.Af * (Tf - val) / Rth) \
                - 2 * self.v / self.H * (val - self.Tc_in)
        return xprsn

    def CN(self, guess, func, valn, datan, datan1, i_n, in1):
        exn1 = func(guess, datan1, in1)
        exn  = func(valn,  datan, i_n)
        ex = guess - valn - self.tstep / 2 * (exn1 + exn)
        return ex

    def InitialConditions(self, guess):
        '''Here, guess is a 2x1'''
        dfCpf, blah = rhocp_fuel(guess[0])
        dcCpc, blah = rhocp_mod(guess[1])
        Rth   = self.thermalResistance(guess[0], guess[1])
        Tfuel_ex = 1 / dfCpf * (self.om * self.p0 - (guess[0] - guess[1]) / Rth)
        Tcool_ex = 1 / dcCpc / self.Aflow * (self.Af * (guess[0] - guess[1]) / Rth) \
                - 2 * self.v / self.H * (guess[1] - self.Tc_in)
        return Tfuel_ex, Tcool_ex

    def fourPhysics(self, i, x):
        '''Solve all 4 physics at once'''
        f = np.zeros(4)
        rho  = self.reactivity(x[2], x[3], i)
        dfCpf, blah  = rhocp_fuel(x[2])
        dcCpc, blah   = rhocp_mod(x[3])
        Rth  = self.thermalResistance(x[2], x[3])
        f[0] = (rho - self.b) / self.L * x[0] + self.l * x[1]
        f[1] = self.b / self.L * x[0] - self.l * x[1]
        f[2] = 1 / dfCpf * (self.om * x[0] - (x[2] - x[3]) / Rth)
        f[3] = 1 / dcCpc / self.Aflow * (self.Af * (x[2] - x[3]) / Rth) \
                - 2 * self.v / self.H * (x[3] - self.Tc_in)
        return f # not a column or row vector specfically at this point

    def reactivityIVP(self, Tf, Tc, t):
        '''Takes a TIME and not an INDEX like reactivity()'''
        if   t >= self.tstrt and t <= self.Ttrans:
            rho_ext = 0
        elif t > self.Ttrans and t <= 2*self.Ttrans:
            rho_ext = 1.45*self.b*(t/self.Ttrans-1)
        else:
            rho_ext = 1.45*self.b
        rho = rho_ext + self.af * (Tf - self.Tf0) + self.ac * (Tc - self.Tc0)
        return rho

    def fourPhysicsIVP(self, t, x):
        '''Solve all 4 physics at once with a time value'''
        f = np.zeros(4)
        rho  = self.reactivityIVP(x[2], x[3], t)
        dfCpf, blah  = rhocp_fuel(x[2])
        dcCpc, blah   = rhocp_mod(x[3])
        Rth  = self.thermalResistance(x[2], x[3])
        f[0] = (rho - self.b) / self.L * x[0] + self.l * x[1]
        f[1] = self.b / self.L * x[0] - self.l * x[1]
        f[2] = 1 / dfCpf * (self.om * x[0] - (x[2] - x[3]) / Rth)
        f[3] = 1 / dcCpc / self.Aflow * (self.Af * (x[2] - x[3]) / Rth) \
                - 2 * self.v / self.H * (x[3] - self.Tc_in)
        return f # not a column or row vector specfically at this point


# End of class OS ###################################################################

def maxDiff(data_old, data):
    difflist = np.zeros(4)
    difflist[0] = np.abs(data[0] - data_old[0]) / data_old[0] # p
    difflist[1] = np.abs(data[1] - data_old[1]) / data_old[1] # c
    difflist[2] = np.abs(data[2] - data_old[2]) / data_old[2] # Tf
    difflist[3] = np.abs(data[3] - data_old[3]) / data_old[3] # Tc
    return np.max(difflist)

def line(conv_order, xsmall, xbig, ysmall):
    xvals = np.zeros(2)
    yvals = np.zeros(2)
    xvals[0] = xsmall
    xvals[1] = xbig
    yvals[0] = ysmall
    yvals[1] = ysmall * (xbig / xsmall) ** conv_order
    return xvals, yvals

class Solvers():
    def __init__(self, tstep):
        self.os = OS(tstep)

    def CN(self, guess, func, datan, i_n, in1):
        exn1 = func(in1, guess)
        exn  = func(i_n, datan)
        ex = guess - datan - self.os.tstep / 2 * (exn1 + exn)
        return ex

    def perturbations(self, X):
        b = np.sqrt(2.220446049250313e-16)
        i = 0
        e = np.zeros(len(X))
        for val in X:
            # e[i] = (1 + val) * b
            e[i] = b * np.max((np.abs(val), 1)) * np.sign(val)
            if e[i] == 0.0:
                e[i] = b
            i += 1
        return e

    def Jacobian(self, Xn, X, i_n, in1):
        e = self.perturbations(X)
        J = np.zeros((len(X),len(X)))
        for j in range(len(X)):
            Xp = X.tolist()
            Xm = X.tolist()
            Xp[j] += e[j]
            Xm[j] -= e[j]
            J[0:4,j] = (self.CN(Xp, self.os.fourPhysics, Xn, i_n, in1) -
                        self.CN(Xm, self.os.fourPhysics, Xn, i_n, in1)) / 2 / e[j]
        return J

    def RHS(self, Xn, X, i_n, in1):
        F = self.CN(X, self.os.fourPhysics, Xn, i_n, in1)
        return F

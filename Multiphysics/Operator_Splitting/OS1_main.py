# NUEN 618HW_2: Neutronics, hydraulics, and heat conduction with CN scheme source
from OS1_utils import OS
from OS1_utils import maxDiff
from OS1_utils import line

import copy
from scipy.optimize import fsolve
# from mpltools import annotation
import numpy as np
import matplotlib.pyplot as plt

# plt.rc('text', usetex=True)
# plt.rcParams['text.usetex'] = True
# plt.rc('text.latex.exe')#, preamble=r'\usepackage{amsmath}')
          # \usepackage{foo-name} `...')

# IMPORTANT ###########################################
#######################################################
# TWO data objects now
# data is a list with with 5 entries;
# datan[0] = pn
# datan[1] = cn
# datan[2] = Tfn
# ddtan[3] = Tcn
# datan[4] = in where i is the time list index

# datan1[0] = pn1
# datan1[1] = cn1
# datan1[2] = Tfn1
# ddtan1[3] = Tcn1
# datan1[4] = in1 where i is the time list index

# dn = data object at tn
# dn1 = data object at tn1
#######################################################
#######################################################

# Inputs ##############################################
# tstep      = 0.001
# physiclist = ['tf', 'tc', 'rk']
# stagger    = 1 # 0 for simultaneous
# iterate    = 1 # 0 for no iteration
# tol        = 1e-10
#######################################################
# if stagger != 0 and stagger != 1:
#     raise ValueError('stagger must be boolean')
# if iterate != 0 and iterate != 1:
#     raise ValueError('iterate must be boolean')

def solve(tstep, physiclist, stagger, iterate, tol):
    os = OS(tstep)
    IC = fsolve(os.InitialConditions, (1000,400)) # NEED THIS!!!!!
    print(IC)
    os.Tf0 = IC[0]
    os.Tc0 = IC[1]
    # First value of each soln vec is the respective IC
    os.p.append(os.p0)
    os.c.append(os.c0)
    os.Tf.append(os.Tf0)
    os.Tc.append(os.Tc0)

    dn    = np.zeros(4)
    tmp   = np.zeros(4)

    dn[0] = os.p0
    dn[1] = os.c0
    dn[2] = os.Tf0
    dn[3] = os.Tc0

    dn1 = copy.copy(dn) # This is the dumbest thing, DONT set a vector = another vector,
    # dn1[4] = 1  # the equality works in both directions for vectors, use v1 = v2[:]
    itlist = []
    # print('time = {}'.format(0.0))
    # print(dn)
    for i in range(1, len(os.time)):
        # if os.time[i] > 0.12:
        #     break
        # print('time = {}'.format(os.time[i]))
        # print('rho = {}'.format(os.reactivity(dn[2], dn[3], i)))
        # print('rho_ext = {}'.format(os.rho_ext[i]))
        d_old = copy.copy(dn1) # Dont care about time index for convergence testing
        diff = tol + 1
        it_num = 0
        while diff >= tol:
            # print('iteration within time step = {}'.format(it_num))
            for physic in physiclist:
                if physic == 'rk':
                    rk_sol = fsolve(os.CN, [dn1[0], dn1[1]],
                                    args=(os.reactorKinetics, [dn[0], dn[1]], dn, dn1, i-1, i))
                    if not stagger:
                        tmp[0] = rk_sol[0] # If simultaneous, update dn1 only after all physics
                        tmp[1] = rk_sol[1] # have been solved
                    else:
                        dn1[0] = rk_sol[0]
                        dn1[1] = rk_sol[1]
                        # print(dn1)
                elif physic == 'tf':
                    tf_sol = np.asscalar(fsolve(os.CN, dn1[2],
                                                args=(os.fuelTemperature, dn[2], dn, dn1, i-1, i)))
                    if not stagger:
                        tmp[2] = tf_sol
                    else:
                        dn1[2] = tf_sol
                        # print(dn1)
                elif physic == 'tc':
                    tc_sol = np.asscalar(fsolve(os.CN, dn1[3],
                                                args=(os.coolantTemperature, dn[3], dn, dn1, i-1, i)))
                    if not stagger:
                        tmp[3] = tc_sol
                    else:
                        dn1[3] = tc_sol
                        # print(dn1)
                else:
                    raise ValueError('physiclist must contain either "rk", "tf", or "tc".')

            if not stagger:
                dn1 = copy.copy(tmp)
            if not iterate:
                break
            else:
                diff = maxDiff(d_old, dn1)
                d_old = copy.copy(dn1)
                it_num += 1
                # Dont need to update anything here right? All updates to dn1 should be
                # happening above
        itlist.append(it_num)
        os.p.append(dn1[0])
        os.c.append(dn1[1])
        os.Tf.append(dn1[2])
        os.Tc.append(dn1[3])
        dn = copy.copy(dn1)
        # dn1[4] += 1
    itlist.insert(0, 0)
    # return this for power plotting
    return os.time, os.p, itlist
    # return this stuff for convergence
    # return os.p0, np.max(os.p)

# os = OS(0.001)
# Rth = os.reactorKinetics([1e7, 5000], [77, 700, 400, 0])
# IC = fsolve(os.InitialConditions, (1000,400)) # NEED THIS!!!!!
# print(IC)
# os.Tf0 = IC[0]
# os.Tc0 = IC[1]
# print(os.thermalResistance(1000,400))
# print('h = {}'.format(os.convectiveHeat(300)))
# print('Tf = {}'.format(os.coolantTemperature(300, [[10000, 0], 400, 300])))
# p0, max, blah = solve(1e-3, ['rk', 'tf', 'tc'], 1, 1, 1e-8)

# refstep, refmax = solve(1e-6, ['tf', 'tc', 'rk'], 1, 1, 1e-7)

# Convergence plotting ##########################################################################
# refmax = 7036502.687293672 # tstep = 1e-6 and tol = 1e-9
# tlist = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
# errorlist = []
# elist2 = []
# for i in range(len(tlist)):
#     # print(i)
#     p0, max = solve(tlist[i], ['tf', 'tc', 'rk'], 0, 1, 1e-7)
#     p02, max2 = solve(tlist[i], ['tf', 'tc', 'rk'], 0, 0, 1e-7)
#     errorlist.append(np.abs(max - refmax)/p0)
#     elist2.append(np.abs(max2 - refmax)/p0)
# fig, ax = plt.subplots()
# x1, y1 = line(1, tlist[-1], tlist[0], elist2[-1])
# ax.loglog(x1, y1, 'k-.', label='1st order', linewidth=0.75)
# x2, y2 = line(2, tlist[-1], tlist[0], errorlist[-1])
# ax.loglog(x2, y2, 'k', label='2nd order', linewidth=0.75)
# ax.loglog(tlist, elist2, '.', label='Not Iterated')
# ax.loglog(tlist, errorlist, '.', label='Iterated')
#
# ax.set_xlabel(r'$\Delta t$ (sec)')
# ax.set_ylabel(r'$\frac{\mathrm{abs}\left(||p||_{L^{\infty}} - ||p_{\mathrm{ref}}||_{L^{\infty}}\right)}{p_0}$')
# ax.legend()
# ax.grid()
# fig.savefig('trial.pdf')
# fig.savefig('simultaneous.pdf')
#################################################################################################

# Power plotting ################################################################################
tlist = [1e-3]# ,5e-4, 1e-4, 5e-5]
for i in range(len(tlist)):
    fig, ax = plt.subplots(2, 1, tight_layout=True)
    # fig2, ax2 = plt.subplots()
    time, p, itlist = solve(tlist[i], ['tf', 'tc', 'rk'], 0, 0, 1e-7)
    ax[0].semilogy(time, p, label='Simul, no it.')
    time2, p2, itlist2 = solve(tlist[i], ['tf', 'tc', 'rk'], 0, 1, 1e-7)
    ax[0].semilogy(time2, p2, label='Simul, with it.')
    ax[1].plot(time2, itlist2, label='Simultaneous')

    time3, p3, itlist3 = solve(tlist[i], ['tf', 'tc', 'rk'], 1, 0, 1e-7)
    ax[0].semilogy(time3, p3, label='Stagger, no it.')
    time4, p4, itlist4 = solve(tlist[i], ['tf', 'tc', 'rk'], 1, 1, 1e-7)
    ax[0].semilogy(time4, p4, label='Stagger, with it.')
    ax[1].plot(time4, itlist4, label='Staggered')
    ax[0].grid()
    ax[0].legend()
    # ax2.grid()
    ax[1].legend()
    ax[0].set_xlabel('t (sec)')
    ax[0].set_ylabel('p (W)')
    ax[1].set_xlabel('t (sec)')
    ax[1].set_ylabel('# Iterations per time step')

    fig.savefig('powertest.pdf')
    # fig.savefig('dub{}.pdf'.format(tlist[i]))
################################################################################################
# print('peak time = {}'.format(os.time[os.p.index(np.max(os.p))]))

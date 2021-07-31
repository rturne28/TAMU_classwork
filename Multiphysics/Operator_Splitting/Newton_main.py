# NUEN 618HW_3: Solving previous HW with Newton's method and GMRES
from OS1_utils import OS
from OS1_utils import Solvers
from OS1_utils import maxDiff
from OS1_utils import line
from OS1_main import solve

from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import gmres
import copy
import numpy as np
import matplotlib.pyplot as plt
import time

def Newton(tstep, tolerance, eval_analytic=True, eval_Newton=True):
    s = Solvers(tstep)
    tol = tolerance
    IC = fsolve(s.os.InitialConditions, (1000,400)) # NEED THIS!!!!!
    s.os.Tf0 = IC[0]
    s.os.Tc0 = IC[1]
    s.os.p.append(s.os.p0)
    s.os.c.append(s.os.c0)
    s.os.Tf.append(s.os.Tf0)
    s.os.Tc.append(s.os.Tc0)

    X = np.zeros(4)
    X[0] = s.os.p0
    X[1] = s.os.c0
    X[2] = s.os.Tf0
    X[3] = s.os.Tc0
    Xold = copy.copy(X)

    nrm = 100
    itlist = []

    for i in range(1, len(s.os.time)):
        if eval_analytic and s.os.time[i] > 0.12:
            break
        nrm = 100
        it_num = 0
        Xstart = copy.copy(X)
        while nrm > tol:
            J = s.Jacobian(Xstart, X, i-1, i)
            F = s.RHS(Xstart, X, i-1, i)
            if eval_Newton:
                dX = np.linalg.solve(J, -F)
            elif not eval_Newton:
                dX, info = gmres(J, -F)
            X = Xold + dX
            nrm = np.linalg.norm(dX) / np.linalg.norm(X)
            Xold = copy.copy(X)
            it_num += 1
        print('rho = {}'.format(s.os.reactivity(X[2], X[3], i)))
        print(s.os.time[i])
        print(J)
        itlist.append(it_num)
        s.os.p.append(X[0])
        s.os.c.append(X[1])
        s.os.Tf.append(X[2])
        s.os.Tc.append(X[3])
    itlist.insert(0, 0)
    return s.os.p, s.os.time

def fSolve(tstep, tolerance, eval_analytic=True):
    s = Solvers(tstep)
    tol = tolerance
    IC = fsolve(s.os.InitialConditions, (1000,400)) # NEED THIS!!!!!
    s.os.Tf0 = IC[0]
    s.os.Tc0 = IC[1]
    s.os.p.append(s.os.p0)
    s.os.c.append(s.os.c0)
    s.os.Tf.append(s.os.Tf0)
    s.os.Tc.append(s.os.Tc0)
    X = np.zeros(4)
    X[0] = s.os.p0
    X[1] = s.os.c0
    X[2] = s.os.Tf0
    X[3] = s.os.Tc0

    Xold = copy.copy(X)
    itlist = []
    for i in range(1, len(s.os.time)):
        if eval_analytic and s.os.time[i] > 0.12:
            break
        diff = 100
        it_num = 0
        Xstart = copy.copy(X)
        while diff > tol:
            X = fsolve(s.CN, X, args=(s.os.fourPhysics, Xstart, i-1, i))
            diff = maxDiff(Xold, X)
            Xold = copy.copy(X)
            it_num += 1
        itlist.append(it_num)
        s.os.p.append(X[0])
        s.os.c.append(X[1])
        s.os.Tf.append(X[2])
        s.os.Tc.append(X[3])
    itlist.insert(0, 0)
    return s.os.p, s.os.time
################################################################################
################################################################################

def convergence(steplist):
    steplist = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    elist_f = []
    elist_N = []
    elist_g = []
    # ss = Solvers(1e-5)
    # IC = fsolve(ss.os.InitialConditions, (1000,400)) # NEED THIS!!!!!
    # ss.os.Tf0 = IC[0]
    # ss.os.Tc0 = IC[1]
    # X = np.zeros(4)
    # X[0] = ss.os.p0
    # X[1] = ss.os.c0
    # X[2] = ss.os.Tf0
    # X[3] = ss.os.Tc0
    # print(X)
    # sol = solve_ivp(ss.os.fourPhysicsIVP, [ss.os.tstrt, ss.os.tend], X,
    #              max_step=1e-6)
    # refmax = np.max(sol.y[0])
    # fig2, ax2 = plt.subplots()
    # ax2.plot(sol.t, sol.y[0])
    # fig2.savefig('analytical.pdf')
    # print(refmax)
    refmax = 7036502.687293672 # from highly refined solution
    for val in steplist:
        fsolvep, time = fSolve(val, 1e-8, True)
        elist_f.append(np.abs(np.max(fsolvep) - refmax)/fsolvep[0])
        Newtonp, blah = Newton(val, 1e-8, True)
        elist_N.append(np.abs(np.max(Newtonp) - refmax)/Newtonp[0])
        gmresp, blah = Newton(val, 1e-8, True, False)
        elist_g.append(np.abs(np.max(gmresp) - refmax)/gmresp[0])

    fig, ax = plt.subplots()
    x1, y1 = line(2, steplist[-1], steplist[0], elist_f[-1])
    ax.loglog(x1, y1, 'k-.', label='2nd order', linewidth=0.75)
    ax.loglog(steplist, elist_f, '.', label='fsolve')
    ax.set_xlabel(r'$\Delta t$ (sec)')
    ax.set_ylabel(r'$\frac{\mathrm{abs}\left(||p||_{L^{\infty}} - ||p_{\mathrm{ref}}||_{L^{\infty}}\right)}{p_0}$')
    ax.legend()
    ax.grid()
    fig.savefig('fsolveconv.pdf')

    fig, ax = plt.subplots()
    x1, y1 = line(2, steplist[-1], steplist[0], elist_N[-1])
    ax.loglog(x1, y1, 'k-.', label='2nd order', linewidth=0.75)
    ax.loglog(steplist, elist_N, '.', label='Newton')
    ax.set_xlabel(r'$\Delta t$ (sec)')
    ax.set_ylabel(r'$\frac{\mathrm{abs}\left(||p||_{L^{\infty}} - ||p_{\mathrm{ref}}||_{L^{\infty}}\right)}{p_0}$')
    ax.legend()
    ax.grid()
    fig.savefig('Newtonconv.pdf')

    fig, ax = plt.subplots()
    x1, y1 = line(2, steplist[-1], steplist[0], elist_g[-1])
    ax.loglog(x1, y1, 'k-.', label='2nd order', linewidth=0.75)
    ax.loglog(steplist, elist_g, '.', label='Newton (gmres)')
    ax.set_xlabel(r'$\Delta t$ (sec)')
    ax.set_ylabel(r'$\frac{\mathrm{abs}\left(||p||_{L^{\infty}} - ||p_{\mathrm{ref}}||_{L^{\infty}}\right)}{p_0}$')
    ax.legend()
    ax.grid()
    fig.savefig('gmresconv.pdf')

def compare():
    steplist = [1e-4]#5e-3, 1e-3, 5e-4]
    for val in steplist:
        fig, ax = plt.subplots()
        # time, origp, itlist = solve(val, ['tf', 'tc', 'rk'], 1, 1, 1e-8)
        fsolvep, blah = fSolve(val, 1e-8, False)
        # for i in range(len(fsolvep)):
        #     print('time = {}, p = {}'.format(blah[i],fsolvep[i]))
        Newtonp, blah = Newton(val, 1e-8, False)
        gmresp, blah = Newton(val, 1e-8, False, False)
        ax.semilogy(time, fsolvep, label='HW 3 fsolve')
        ax.semilogy(time, Newtonp, label='Newton')
        ax.semilogy(time, gmresp, label='Newton (gmres)')
        ax.semilogy(time, origp, label='HW 2 fsolve')
        ax.set_xlabel('t (sec)')
        ax.set_ylabel('p (W)')
        ax.grid()
        ax.legend()
        # fig.savefig('HW3_{}.pdf'.format(val))
        # print('timestep = {}'.format(val))
        op = np.max(origp)
        fp = np.max(fsolvep)
        Np = np.max(Newtonp)
        gp = np.max(gmresp)
        print(op)
        # print('{} {}'.format(fp, np.abs(fp-op)/op))
        # print('{} {}'.format(Np, np.abs(Np-op)/op))
        # print('{} {}'.format(gp, np.abs(gp-op)/op))


# convergence(steplist)
compare()

import sympy as sy
import numpy as np
import math as m
from PA_4_Utilities import FEM
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import time
start_time = time.time()

def doit():
    prob_list = [0, 1, 2, 3, 4, 5]
    e_list = [6, 11, 21, 41]
    fig_list = ['P0.pdf', 'P1.pdf', 'P2.pdf', 'P3.pdf', 'P4.pdf', 'P5.pdf']
    for pi in range(len(prob_list)):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for i in range(len(e_list)):
            f = FEM(e_list[i], prob_list[pi])
            f.elementize()
            f.fillAsum()
            f.fillF()
            f.handleBCs()
            f.solve()
            f.evalex()
            f.getL2()
            print(f._L2)
            print(f._H1)
            print(f._H2)
            ax1.plot(f._g_quad_vals, f._plotsol, label='N = {}'.format(f._num_e_boundaries-1))
            ax2.semilogy(f._g_quad_vals, np.abs(f._exact - f._plotsol), label='N = {}'.format(f._num_e_boundaries-1))
        ax1.grid()
        ax1.set_ylabel('$u_h(x)$')
        ax1.set_title('FEM Solutions $u_h$')
        ax1.legend()
        ax2.grid()
        ax2.set_xlabel('Beam Location')
        ax2.set_title('Errors $|u-u_h|$')
        fig.savefig(fig_list[pi])

def norms():
    prob_list = [0, 1, 2, 3, 4, 5]
    e_list = [6, 11, 21, 41]
    norm_table = np.zeros((len(e_list)*len(prob_list),5))
    for pi in range(len(prob_list)):
        for i in range(len(e_list)):
            f = FEM(e_list[i], prob_list[pi])
            f.elementize()
            f.fillAsum()
            f.fillF()
            f.handleBCs()
            f.solve()
            f.evalex()
            f.getL2()
            f.getInf()
            norm_table[len(e_list)*pi+i][0] = f._L2
            norm_table[len(e_list)*pi+i][1] = f._H1
            norm_table[len(e_list)*pi+i][2] = f._H2
            norm_table[len(e_list)*pi+i][3] = f._inf
            norm_table[len(e_list)*pi+i][-1] = f._h
    np.set_printoptions(precision=3)
    print(norm_table)

def basis():
    xvals = np.linspace(-1,1,100)
    f = FEM(2, 0)
    onevals = np.zeros((100,1))
    twovals = np.zeros((100,1))
    threevals = np.zeros((100,1))
    fourvals = np.zeros((100,1))

    for i in range(100):
        onevals[i] = f.phi_(xvals[i])[0]
        twovals[i] = f.phi_(xvals[i])[1]
        threevals[i] = f.phi_(xvals[i])[2]
        fourvals[i] = f.phi_(xvals[i])[3]
    plt.plot(xvals, onevals, label='$\phi_0$')
    plt.plot(xvals, twovals, label='$\phi_1$')
    plt.plot(xvals, threevals, label='$\phi_2$')
    plt.plot(xvals, fourvals, label='$\phi_3$')
    plt.grid()
    plt.legend()
    plt.xlabel('s')
    plt.title('Cubic Hermite Basis Functions')
    plt.savefig('polys.pdf')
    plt.show()

#doit()
# norms()
basis()

# print(f._h)
# print(f._3mass)
# print(f._3stiff)
# print(f._3bend)

# print(f._U)
# print(f._plotsol)

# plt.plot(f._g_quad_vals, np.abs(f._exact - f._plotsol), 'b*--')
# plt.plot(f._g_quad_vals, f._exact, marker='*', label='exact')
# plt.plot(f._g_quad_vals, f._plotsol, marker='.', label='mysoln')
# plt.legend()
# plt.show()
# plt.savefig('test.pdf')
#f.getL2()
# print(f._U)
# print(f._L2)

print("--- %s seconds ---" % (time.time() - start_time))

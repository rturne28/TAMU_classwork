# Steady state, nonlinear, 1D heat conduction with linear finite elements
from HW4_utils import Mesh
from HW4_utils import ReferenceElement
from HW4_utils import Newton

import matplotlib.pyplot as plt
import csv

def execute():
    '''Overlays Newton with fsolve, Newton with gmres and Jacobian, Newton
       with JFNK gmres, MOOSE output, and the analytic solution for 3 spatial
       discretizations as well as outputs the number of both linear and
       nonlinear iterations'''

    delta_x = [0.018, 0.009, 0.0045]
    refelem = ReferenceElement()
    for i in range(len(delta_x)):
        Moosesol = []
        xvals = []
        with open('{}.csv'.format(i)) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            next(readCSV)
            for row in readCSV:
                xvals.append(float(row[0]))
                Moosesol.append(float(row[4]))
        mesh = Mesh(delta_x[i])
        n = Newton(mesh, refelem)
        n.analytic(mesh.vertices)
        n.solve()
        print('@@@@@@@@@@@@@@@@')
        ngmres = Newton(mesh, refelem, True)
        ngmres.solve()
        print('@@@@@@@@@@@@@@@@')
        njfnk  = Newton(mesh, refelem, True, True)
        njfnk.solve()
        print('@@@@@@@@@@@@@@@@')
        fig, ax = plt.subplots()
        ax.plot(mesh.vertices, n.T, '.', label='Newton with fsolve')
        ax.plot(mesh.vertices, ngmres.T, '.', label='Newton with gmres and Jac')
        ax.plot(mesh.vertices, njfnk.T, '.', label='Newton with gmres and JFNK')
        ax.plot(xvals, Moosesol, '.', label='Moose')
        ax.plot(mesh.vertices, n.Ttrue, '.', label='Analytic')
        ax.set_xlabel('x')
        ax.set_ylabel('T(x)')
        ax.grid()
        ax.legend()
        fig.savefig('{}.pdf'.format(mesh.numpnts-1))
        print('##########################################')
        print('##########################################')

execute()

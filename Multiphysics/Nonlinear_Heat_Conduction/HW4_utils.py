# Steady state, nonlinear, 1D heat conduction with linear finite elements
import numpy as np
import sympy as sp
import copy
import mpmath
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
import sys

class Mesh:
    def __init__(self, stepsize):
        self.xstrt    = 0
        self.xend     = 0.45
        self.xstep    = stepsize # must be a factor of 0.45!!!
        self.numpnts  = int((self.xend - self.xstrt)/self.xstep+1)
        self.vertices = np.linspace(self.xstrt, self.xend, self.numpnts)
        self.nodeIDs  = range(self.numpnts)
        print(self.numpnts-1)
        # self.node2coord = {}
        # for i in range(len(self.nodeIDs)):
        #     self.node2coord.update({self.nodeIDs[i]:self.vertices[i]})

class ReferenceElement:
    def __init__(self):
        '''For 1D linear finite elements at the moment'''
        self.s, self.weights = np.polynomial.legendre.leggauss(2)
        self.b0  = []
        self.b1  = []
        self.db0 = []
        self.db1 = []
        for qp in self.s:
            self.b0.append((1-qp)/2)
            self.b1.append((1+qp)/2)
            self.db0.append(-0.5)
            self.db1.append(0.5)

# class Cell:
#     def __init__(self, nodeIDs, vertices):
#         self.nodeIDs = nodeIDs
#         self.vertices = vertices
        #self.area = area # not needed until 2D
        # could have element ID here as well...

class Newton:
    def __init__(self, mesh_, relem_, gmrs=False, action=False): # these are already instantiated objects
        self.mesh   = mesh_
        self.relem  = relem_
        self.tol    = 1e-8
        self.q      = 3e4
        self.Tright = 300
        self.T      = np.zeros(len(self.mesh.nodeIDs))

        self.gmresbool  = gmrs
        self.action     = action
        try:
            if not self.gmresbool and self.action:
                raise ValueError("gmresbool and action should not both be True")
        except ValueError as err:
            print(err.args[0])
            sys.exit(1)

        self.Ttrue = []

    def conduction(self, b0, b1, Tvals):
        '''Evaluate k at a quadrature point'''
        k = 1.5 + 2510 / (215 + Tvals[0]*b0 + Tvals[1]*b1)
        return k

    def diffusionKernel(self, Tvals):
        '''Evaluate the integral of the diffusion term, this term is positive
           as it is on the left hand side and becomes positive after IBP;
           basis functions have already been evaluated at the quad points in
           the refelem object'''
        diff_sol = np.zeros(2)
        for i in range(len(self.relem.s)):
            k = self.conduction(self.relem.b0[i], self.relem.b1[i], Tvals)
            # This is too hardcoded, needs to be more general for 2D
            diff_sol[0] += self.relem.weights[i] * k * \
                           (Tvals[0]*self.relem.db0[i]**2 + Tvals[1]*self.relem.db0[i]*self.relem.db1[i])
            diff_sol[1] += self.relem.weights[i] * k * \
                           (Tvals[0]*self.relem.db1[i]*self.relem.db0[i] + Tvals[1]*self.relem.db1[i]**2)
        diff_sol = 2 / self.mesh.xstep * diff_sol
        return diff_sol

    def sourceKernel(self):
        src_sol = np.zeros(2)
        for i in range(len(self.relem.s)):
            src_sol[0] += self.relem.weights[i] * self.q * self.relem.b0[i]
            src_sol[1] += self.relem.weights[i] * self.q * self.relem.b1[i]
        src_sol = self.mesh.xstep / 2 * src_sol
        return src_sol

    def elementalResidual(self, Tvals):
        '''Wish I could pass the element object, but I would rather not
           pass either the entire solution vector or the Tsnip AND the element'''
        sol = self.diffusionKernel(Tvals) - self.sourceKernel()
        return sol # vector of length 2, deal with orintation later

    def assembleF(self, T):
        '''Creates global residual of length # nodes with a given input vector
           by assembling the 2x1 solution on each element in the global F'''
        F = np.zeros(len(T))
        for node in self.mesh.nodeIDs[:-1]:
            # cell = Cell([node, node+1], # This is hacky....
            #             [self.mesh.vertices[node], self.mesh.vertices[node+1]])
            # Tsnip = [T[cell.nodeIDs[0]], T[cell.nodeIDs[1]]]
            Tsnip = [T[node],T[node+1]]
            elemsol = self.elementalResidual(Tsnip)
            # THIS INDEXING IS NASTY, NEED BETTER WAY FOR 2D ##########
            F[node:node+2] += elemsol # still just a list of shape (2,)
            ###########################################################
        # Hardcoded boundary condition
        F[-1] = T[-1] - self.Tright
        return F

    def Jcolumns(self, T, p, action):
        '''Returns a vector if not action and returns a linear operator if action'''
        '''Be careful '''
        b = 1.4901161193847656e-08
        # e = b * np.linalg.norm(T)
        def actionJ(v):
            if action:
                sum = 0
                for val in T:
                    sum += 1 + val
                nrm = np.linalg.norm(v)
                if nrm == 0:
                    nrm = 1
                e = b / len(T) / nrm * sum
            else:
                e = (1 + np.linalg.norm(T)) * b
            DF = (self.assembleF(T + e*v) - self.assembleF(T)) / e
            return DF
        if not action:
            # print('not using action')
            return actionJ(p)
        else:
            # print('action is being used')
            return LinearOperator((len(T),len(T)), matvec=actionJ)

    def assembleJ(self, T):
        J = np.zeros((len(T),len(T)))
        DF = np.zeros((len(T),3))
        P = np.zeros((len(T),3))
        for i in range(3):
            p = np.zeros(len(T))
            p[i::3] = 1     # obj[start:stop:step]
            P[0:len(T),i] = p
            DF[0:len(T),i] = self.Jcolumns(T, p, False) # Evaluates the 3 columns of values that make up the Jacobian
        j = 1
        for i in range(1,len(T)-1):
            if j > 2:
                j = 0
            J[i-1:i+2,i] = DF[i-1:i+2,j]
            j += 1
        if j == 3:
            j = 0
        J[0,0] = DF[0,0]    # first element
        J[1,0] = DF[1,0]    # element below first element
        J[-2,-1] = DF[-2,j] # element above bottom corner
        J[-1,-1] = DF[-1,j] # bottom corner element
        return J

    def solve(self):
        Told = self.Tright * np.ones(len(self.mesh.nodeIDs))
        T = copy.copy(Told)
        nrm = self.tol + 1
        it = 1
        while nrm > self.tol:
            print('Nonlinear iteration {}'.format(it))
            F = self.assembleF(T)
            if not self.action:
                J = self.assembleJ(T)
            else:
                J = self.Jcolumns(T, 3, True)
            if not self.gmresbool and not self.action:
                dT = np.linalg.solve(J, -F)
            else:
                cntr = Counter(printout=False)
                dT, info = gmres(J, -F, callback=cntr.count)#maxiter=50,restart=50
                print('Linear Iterations = {}'.format(cntr.its))
            T = Told + dT
            nrm = np.linalg.norm(dT) / np.linalg.norm(T)
            Told = copy.copy(T)
            it += 1
        self.T = copy.copy(T)

    def analytic(self, xvals):
        # USING SYMPY TO GET ANALYTIC SOLUTION ####################################
        # T = Function('T')
        # Tr, x, q, L = symbols('Tr, x, q, L')
        # eq = solve(1.5*(T(x)-Tr) + 2510*log((215+T(x))/(215+Tr)) + q/2*(x**2-L**2), T(x))
        # print(eq)
        ###########################################################################
        for x in xvals:
            lamb = mpmath.lambertw(0.000679545158442411*(self.Tright + 215.0) *
                                   np.exp(0.000199203187250996*self.mesh.xend**2 *
                                   self.q + 0.000597609561752988*self.Tright -
                                   0.000199203187250996*self.q*x**2))
            self.Ttrue.append(1673.33333333333 * lamb - 215.0)
#### END NEWTON CLASS #############################################################
###################################################################################

class Counter:
    def __init__(self, printout=True):
        self.its = 0
        self.printout = printout

    def count(self, rk):
        self.its += 1
        if self.printout:
            print('Linear Iteration {}, rk = {}'.format(self.its, rk))

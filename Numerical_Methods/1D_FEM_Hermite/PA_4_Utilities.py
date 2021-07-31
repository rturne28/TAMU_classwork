
# s = symbols('s')
# phi_0 = 2*(s/2+1/2)**3 - 3*(s/2+1/2)**2 + 1
# psi_0 = (s/2+1/2)**3 - 2*(s/2+1/2)**2 + (s/2+1/2)
# g = phi_0.subs(s, 0.5)
# print(g)

# Some description
import sympy as sy
import numpy as np
import math as m
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


class FEM:
    def __init__(self, num_e_boundaries, prob_switch):
        self._num_e_boundaries = num_e_boundaries
        self._prob = prob_switch # 0 for Dirichlet, 1 for Neumann, 2 for Robin

        self._leftBound = 0
        self._rightBound = 1
        self._e_size = 4
        self._l = self._rightBound - self._leftBound
        self._h = self._l/(self._num_e_boundaries-1)
        self._xvals = np.linspace(self._leftBound, self._rightBound, self._num_e_boundaries)
        self._g_size = (self._num_e_boundaries-1)*2+2

        self._k = 0
        self._p = 0
        self._g = 0

        if self._prob == 0 or self._prob == 1 or self._prob ==  3: #1.1
            self._k = 1
            self._p = 0
            self._g = 0
        elif self._prob == 2 or self._prob == 4:
            self._k = 1/(4*m.pi**4)
            self._p = 0
            self._g = 1
        elif self._prob == 5:
            self._k = 1
            self._p = 0
            self._g = 0
        else:
            raise ValueError('Problem Switch must be 0 thru 5.')

        self._3mass = np.zeros((self._e_size, self._e_size))
        self._3stiff = np.zeros((self._e_size, self._e_size))
        self._3bend = np.zeros((self._e_size, self._e_size))

        self._Asum = np.zeros((self._g_size,self._g_size))
        self._F = np.zeros((self._g_size,1))
        # this should NOT be used for plotting
        self._U = np.zeros((self._g_size,1)) # solution vector, need function that evaluates this
        # norm function will fill these as it evaluate the norms
        self._plotsol = np.zeros((self._num_e_boundaries, 1))
        self._exact = np.zeros((self._num_e_boundaries, 1))# this is only for plotting and max norm, not L2 and H1
        self._g_quad_vals = np.zeros((self._num_e_boundaries,1))

        self._L2 = 0
        self._H1 = 0
        self._H2 = 0
        self._inf = 0

    def phi_(self, point):
        s = sy.symbols('s')
        eq0 = ((1-s)/2)**2 * (2+s)
        eq1 = ((1-s)/2)**2 * ((1+s)*self._h/2)
        eq2 = ((1+s)/2)**2 * (2-s)
        eq3 = ((1+s)/2)**2 * ((s-1)*self._h/2)
        f0 = eq0.subs(s, point)
        f1 = eq1.subs(s, point)
        f2 = eq2.subs(s, point)
        f3 = eq3.subs(s, point)
        return f0, f1, f2, f3

    def phi_p(self, point):
        s = sy.symbols('s')
        eq0 = ((1-s)/2)**2 * (2+s)
        eq1 = ((1-s)/2)**2 * ((1+s)*self._h/2)
        eq2 = ((1+s)/2)**2 * (2-s)
        eq3 = ((1+s)/2)**2 * ((s-1)*self._h/2)
        f0 = sy.diff(eq0, s).subs(s, point)
        f1 = sy.diff(eq1, s).subs(s, point)
        f2 = sy.diff(eq2, s).subs(s, point)
        f3 = sy.diff(eq3, s).subs(s, point)
        return f0, f1, f2, f3

    def phi_pp(self, point):
        s = sy.symbols('s')
        eq0 = ((1-s)/2)**2 * (2+s)
        eq1 = ((1-s)/2)**2 * ((1+s)*self._h/2)
        eq2 = ((1+s)/2)**2 * (2-s)
        eq3 = ((1+s)/2)**2 * ((s-1)*self._h/2)
        f0 = sy.diff(eq0, s, 2).subs(s, point)
        f1 = sy.diff(eq1, s, 2).subs(s, point)
        f2 = sy.diff(eq2, s, 2).subs(s, point)
        f3 = sy.diff(eq3, s, 2).subs(s, point)
        return f0, f1, f2, f3

    # def elementize(self):
    #     s, weights = np.polynomial.legendre.leggauss(6)
    #
    #     for j in range(4):
    #         for k in range(4):
    #             for i in range(size.s)
    #                 self._3mass[j][k] = weights[i] * self.phi_(s[i])[] * self.phi_(s[i])[]
    #     for j in range(4):
    #         for k in range(4):
    #             self._3stiff[j][k] = sum([weights[l] * phi_s[j][l] * phi_s[k][l] for l in range(len(s.tolist()))])
    #     for j in range(4):
    #         for k in range(4):
    #             self._3bend[j][k] = sum([weights[l] * phi_b[j][l] * phi_b[k][l] for l in range(len(s.tolist()))])

    def elementize(self):
        s, weights = np.polynomial.legendre.leggauss(6)
        phi_m = np.zeros((self._e_size, len(s)))
        phi_s = np.zeros((self._e_size, len(s)))
        phi_b = np.zeros((self._e_size, len(s)))
        for i in range(len(s.tolist())):
            phi_m[0][i] = self.phi_(s[i])[0]
            phi_m[1][i] = self.phi_(s[i])[1]
            phi_m[2][i] = self.phi_(s[i])[2]
            phi_m[3][i] = self.phi_(s[i])[3]
            phi_s[0][i] = self.phi_p(s[i])[0]
            phi_s[1][i] = self.phi_p(s[i])[1]
            phi_s[2][i] = self.phi_p(s[i])[2]
            phi_s[3][i] = self.phi_p(s[i])[3]
            phi_b[0][i] = self.phi_pp(s[i])[0]
            phi_b[1][i] = self.phi_pp(s[i])[1]
            phi_b[2][i] = self.phi_pp(s[i])[2]
            phi_b[3][i] = self.phi_pp(s[i])[3]
        for j in range(4):
            for k in range(4):
                self._3mass[j][k] = sum([weights[l] * phi_m[j][l] * phi_m[k][l] for l in range(len(s.tolist()))])
        for j in range(4):
            for k in range(4):
                self._3stiff[j][k] = sum([weights[l] * phi_s[j][l] * phi_s[k][l] for l in range(len(s.tolist()))])
        for j in range(4):
            for k in range(4):
                self._3bend[j][k] = sum([weights[l] * phi_b[j][l] * phi_b[k][l] for l in range(len(s.tolist()))])

    def globalize(self, mass_or_stiff):
        '''Builds global matrices from elemental mass and stiffness matrices'''
        g_matrix = np.zeros((self._g_size, self._g_size))
        for i in range(self._num_e_boundaries-1):
            g_matrix[(2*i):(2*i+4), (2*i):(2*i+4)] += mass_or_stiff
        return g_matrix

    def fillAsum(self):
        gm_matrix = self.globalize(self._3mass)
        gs_matrix = self.globalize(self._3stiff)
        gb_matrix = self.globalize(self._3bend)

        self._Asum = self._h/2 * ((2/self._h)**4*self._k*gb_matrix
                                + (2/self._h)**2*self._p*gs_matrix
                                               + self._g*gm_matrix)
        np.set_printoptions(precision=1)
        # print(self._Asum)

        # mat = self._h/2 * ((2/self._h)**2*gs_matrix + self._S/self._D*gm_matrix)


#######################################################################
    def fillF(self):
        '''Don't use member variable so it won't be overwritten'''
        '''function is here so it can take advantage of the member variables'''
        s, weights = np.polynomial.legendre.leggauss(6)
        for ebi in range(self._num_e_boundaries-1):
            for oi in range(self._e_size):
                delta = np.zeros((s.size, 1))
                for i in range(s.size):
                    x = (s[i] + 1)/2 * self._h + self._xvals[ebi]
                    if self._prob == 0 or self._prob == 3:
                        f = m.sin(m.pi*x)
                    elif self._prob == 1:
                        f = 60*x
                    elif self._prob == 2 or self._prob == 4:
                        f = 1
                    elif self._prob == 5:
                        f = 70*x**2 - 5
                    else:
                        raise ValueError('Problem Switch must be 0 thru 5.')

                    phi = self.phi_(s[i])[oi]
                    delta[i] = f * self._h/2 * phi * weights[i]
                self._F[ebi*(self._e_size-2)+oi] += sum(delta)
###############################################################################
    def singleDirichlet(self):
        '''Applies a homogeneous Dirichlet BC to the left AND right boundary'''
        self._Asum[0] = 0
        self._Asum[-2] = 0
        self._Asum[:][0] = 0
        self._Asum[:][-2] = 0
        self._Asum[0][0] = 1
        self._Asum[-2][-2] = 1

        self._F[0] = 0
        self._F[-2] = 0

    def doubleDirichlet(self):
        self._Asum[0] = 0
        self._Asum[1] = 0
        self._Asum[-2] = 0
        self._Asum[-1] = 0
        self._Asum[:][0] = 0
        self._Asum[:][1] = 0
        self._Asum[:][-2] = 0
        self._Asum[:][-1] = 0
        self._Asum[0][0] = 1
        self._Asum[1][1] = 1
        self._Asum[-2][-2] = 1
        self._Asum[-1][-1] = 1

        self._F[0] = 0
        self._F[1] = 0
        self._F[-2] = 0
        self._F[-1] = 0

    def handleBCs(self):
        if self._prob == 0 or self._prob == 1 or self._prob == 2:
            self.singleDirichlet()
        elif self._prob == 3 or self._prob == 4 or self._prob == 5:
            self.doubleDirichlet()
        else:
            raise ValueError('Problem Switch must be 0 thru 5.')

    def solve(self):
        # Use this one for scripts outside of file
        self._U = np.linalg.solve(self._Asum, self._F)
        # Use this one for code inside this file
        # self._U = np.linalg.solve(fem._Asum, fem._F)

    def eval_exact_discrete(self, qp, x_left):
        '''this function takes as input a QUADRATURE POINT AND the current'''
        '''the X LOCATION of the left element boundary - doing it this way'''
        '''in order to use sympy for derivatives'''
        s, x = sy.symbols('s x')
        x = (s+1)/2 * self._h + x_left
        A = 1/(m.exp(m.pi)-1)
        B = - m.exp(m.pi)/(m.exp(m.pi)-1)
        if self._prob == 0:
            u = 1/(m.pi**4)*sy.sin(m.pi*x)#m.pi*x
        elif self._prob == 1:
            u = x/6*(7-10*x**2 + 3*x**4)
        elif self._prob == 2:
            u = 1+sy.cos(m.pi*x)*(A*sy.exp(m.pi*x)+B*sy.exp(-m.pi*x))
        elif self._prob == 3:
            u = x**2/m.pi**3 - x/m.pi**3 + sy.sin(m.pi*x)/m.pi**4
        elif self._prob == 4:
            u = 1+sy.cos(m.pi*x)*(A*sy.exp(m.pi*x)+B*sy.exp(-m.pi*x)) \
                 -sy.sin(m.pi*x)*(A*sy.exp(m.pi*x)-B*sy.exp(-m.pi*x))
        elif self._prob == 5:
            u = 1/72*(x-1)**2*x**2*(14*x**2+28*x+27)
        else:
            raise ValueError('Problem Switch must be 0 thru 5.')

        exact_ = u.subs(s, qp)
        exact_p = sy.diff(u, s).subs(s, qp)
        exact_pp = sy.diff(u, s, 2).subs(s, qp)
        return exact_, exact_p, exact_pp

    def evalex(self):
        for i in range(len(self._xvals)):
            self._exact[i] = self.eval_exact_discrete(-1, self._xvals[i])[0]
            self._plotsol[i] = self._U[2*i]
    #
    # def eval_exactN(self, g_point_t):
    #     '''evaluates the exact neumann soln (W'(t) = 0) and the derivative at a point'''
    #     '''needs the global point from 0 to 1, aka t)'''
    #
    #     g = self._gamma
    #     A = self._bigA
    #     t = g_point_t
    #     tp = self._h / self._l / 2
    #     w = A/g*(-t**2+t-2/g+1/(g*m.cosh(m.sqrt(g)))*(m.sqrt(g)*m.sinh(m.sqrt(g)*t)+2*m.cosh(m.sqrt(g)*(1-t))))
    #     wp = A/g*(-2*t*tp+tp+1/(g*m.cosh(m.sqrt(g)))*(g*tp*m.cosh(m.sqrt(g)*t)-2*m.sqrt(g)*tp*m.sinh(m.sqrt(g)*(1-t))))
    #     return w, wp

    def getL2(self):
        '''must be called after solving the system'''
        h = self._h
        domainL2 = 0
        domainH1 = 0
        domainH2 = 0
        s, weights = np.polynomial.legendre.leggauss(20) #should this be order * order?
        self._plotsol.resize((s.size*(self._num_e_boundaries-1),1))
        self._exact.resize((s.size*(self._num_e_boundaries-1),1))
        self._g_quad_vals.resize((s.size*(self._num_e_boundaries-1),1))
        for ebi in range(self._num_e_boundaries-1): # element loop
            elementL2 = 0
            elementH1 = 0
            elementH2 = 0
            for i in range(s.size):                # quadrature loop
                mysoln_ = 0
                mysoln_p = 0
                mysoln_pp = 0
                for oi in range(self._e_size):     # inner element nodal loop
                    phi_ = self.phi_(s[i])[oi]
                    phi_p = self.phi_p(s[i])[oi]
                    phi_pp = self.phi_pp(s[i])[oi]
                    mysoln_ += self._U[ebi*(self._e_size-2)+oi]*phi_
                    mysoln_p += self._U[ebi*(self._e_size-2)+oi]*phi_p
                    mysoln_pp += self._U[ebi*(self._e_size-2)+oi]*phi_pp

                #x = (s[i] + 1)/2 * self._h + self._xvals[ebi]
                true_soln_ = self.eval_exact_discrete(s[i], self._xvals[ebi])[0]
                true_soln_p = self.eval_exact_discrete(s[i], self._xvals[ebi])[1]
                true_soln_pp = self.eval_exact_discrete(s[i], self._xvals[ebi])[2]

                self._plotsol[ebi*s.size+i] = mysoln_
                self._exact[ebi*s.size+i] = true_soln_
                self._g_quad_vals[ebi*s.size+i] = (s[i]+1)/2 * h + self._xvals[ebi]
                # if ebi == 1 and i == 2:
                    # print(mysoln_)
                    # print(true_soln_)
                elementL2 += h/2*(mysoln_ - true_soln_)**2 * weights[i]
                elementH1 += h/2*(4/h**2*(mysoln_p - true_soln_p)**2
                             + (mysoln_ - true_soln_)**2) * weights[i]
                elementH2 += h/2*((2/h)**4*(mysoln_pp - true_soln_pp)**2
                             + 4/h**2*(mysoln_p - true_soln_p)**2
                             + (mysoln_ - true_soln_)**2) * weights[i]
            domainL2 += elementL2
            domainH1 += elementH1
            domainH2 += elementH2
        self._L2 = m.sqrt(domainL2)
        self._H1 = m.sqrt(domainH1)
        self._H2 = m.sqrt(domainH2)
        #self._l2 = np.linalg.norm(abs(self._U - self._exact))

    def getInf(self):
        self._inf = np.linalg.norm(abs(self._plotsol - self._exact), np.inf)




# elem_matrices = FEMMatrices()
# elem_matrices.firstOrder() # These actually fill the mass and stiffness member variables of FEMMatrices
# elem_matrices.secondOrder() # This is less than desirable but o well
# elem_matrices.thirdOrder()
# element_bounds = [11, 21, 41]
#
#     #for
# fem = FEM(elem_matrices, 41, 3, 100, 0)
# fem.fillAsum()
# fem.fillF()
# fem.handleBC()
# fem.solve()
#
#
# fem.getL2()
# fem.getH1()
# fem.getInf()
# #print(mysoln)
# print(fem._L2)
# print(fem._H1)
# print(fem._inf)
# plt.plot(fem._nodevals, fem._U, label='mine', marker='*')
# plt.plot(fem._nodevals, fem._exact, label='correct', marker='.')
# plt.legend()
# plt.grid()
#plt.savefig('plot.pdf')
#plt.show()


# h=2
# def phi_0(point):
#     s = symbols('s')
#     eq = ((1-s)/2)**2 * (2+s)
#     f = eq.subs(s, point)
#     fp = diff(eq, s).subs(s, point)
#     fpp = diff(eq, s, 2).subs(s, point)
#     return f, fp, fpp
#
# def phi_1(point,h):
#     s = symbols('s')
#     eq = ((1-s)/2)**2 * ((1+s)*h/2)
#     f = eq.subs(s, point)
#     fp = diff(eq, s).subs(s, point)
#     fpp = diff(eq, s, 2).subs(s, point)
#     return f, fp, fpp
#
# def phi_2(point):
#     s = symbols('s')
#     eq = ((1+s)/2)**2 * (2-s)
#     f = eq.subs(s, point)
#     fp = diff(eq, s).subs(s, point)
#     fpp = diff(eq, s, 2).subs(s, point)
#     return f, fp, fpp
#
# def phi_3(point,h):
#     s = symbols('s')
#     eq = ((1+s)/2)**2 * ((s-1)*h/2)
#     f = eq.subs(s, point)
#     fp = diff(eq, s).subs(s, point)
#     fpp = diff(eq, s, 2).subs(s, point)
#     return f, fp, fpp
#
# # print(f)
#     # for i in range(10000):
# # d, dd, ddd = phi_0(4)
# # p = ddd+ddd**3/.8
# # print(p)
# # print(type(p))
# # t = 0.343
# # print(type(t))
# # print(d)
# # print(dd)
# # print(ddd)
# # print(dd)
# # print(ddd)
# # import numpy as np
# s, weights = np.polynomial.legendre.leggauss(6)
# phi_m = np.zeros((4, len(s)))
# phi_s = np.zeros((4, len(s)))
# phi_b = np.zeros((4, len(s)))
#
# _2mass = np.zeros((4,4))
# _2stiff = np.zeros((4,4))
# _2bend = np.zeros((4,4))
# for i in range(len(s.tolist())):
#     phi_m[0][i] = phi_0(s[i])[0]
#     phi_m[1][i] = phi_1(s[i],h)[0]
#     phi_m[2][i] = phi_2(s[i])[0]
#     phi_m[3][i] = phi_3(s[i],h)[0]
#     phi_s[0][i] = phi_0(s[i])[1]
#     phi_s[1][i] = phi_1(s[i],h)[1]
#     phi_s[2][i] = phi_2(s[i])[1]
#     phi_s[3][i] = phi_3(s[i],h)[1]
#     phi_b[0][i] = phi_0(s[i])[2]
#     phi_b[1][i] = phi_1(s[i],h)[2]
#     phi_b[2][i] = phi_2(s[i])[2]
#     phi_b[3][i] = phi_3(s[i],h)[2]
# for j in range(4):
#     for k in range(4):
#         _2mass[j][k] = sum([weights[l] * phi_m[j][l] * phi_m[k][l] for l in range(len(s.tolist()))])
# for j in range(4):
#     for k in range(4):
#         _2stiff[j][k] = sum([weights[l] * phi_s[j][l] * phi_s[k][l] for l in range(len(s.tolist()))])
# for j in range(4):
#     for k in range(4):
#         _2bend[j][k] = sum([weights[l] * phi_b[j][l] * phi_b[k][l] for l in range(len(s.tolist()))])
#
# print(_2mass)
# print(_2stiff)
# print(_2bend)

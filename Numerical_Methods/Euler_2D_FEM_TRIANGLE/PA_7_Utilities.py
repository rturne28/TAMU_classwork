import numpy as np
import math as m
from quadpy.triangle import Walkington
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import eigs


class TFEM:
    def __init__(self, discretization, num_t_points, problem, timescheme):
        '''Dropping the underscore on member variables'''
        self.nodes = []
        self.edges = []
        self.elems = []
        self.rows = []
        self.cols = []
        self.vals = []

        self.Drows = []
        self.Dcols = []
        self.Dvals = []
        # Time info ############################################################
        self.scheme = timescheme
        # Backwards, unconditionally stable ######################
        if self.scheme == 0:
            self.t_0 = 0
            self.t_end = 3
            self.tpnts = num_t_points
            self.tau =  (self.t_end - self.t_0) / (self.tpnts-1)
            self.tvals = np.linspace(self.t_0, self.t_end, self.tpnts)
        ##########################################################
        # Stable forward #########################################
        elif self.scheme == 1:
            self.tpnts = 1
            self.tau = None
            self.tvals = None
        ##########################################################
        # Unstable forward #######################################
        elif self.scheme == 2:
            self.tpnts = 10
            self.tau = 3/(self.tpnts-1)
            self.tvals = np.linspace(0, 3, self.tpnts)
        ##########################################################
        else:
            raise ValueError('scheme must be either 0 or 1.')
        ########################################################################
        self.readTriangleMesh(discretization)
        self.prob = problem
        if self.prob == 0:
            self.q = 5
            self.g = 0
        elif self.prob == 1 or self.prob == 2:
            self.q = 1
            self.g = 1
        else:
            raise ValueError('Problem Switch must be 0 thru 2.')

        self.order = 5
        self.qpoints = Walkington(self.order).points
        self.qweights = Walkington(self.order).weights * 0.5

        self.gradphi = np.array([[-1, 1, 0],
                                 [-1, 0, 1]])

        self.g_F = np.zeros(len(self.nodes)) #This used to be (self.nodes, 1)
        self.U = np.zeros((self.tpnts, len(self.nodes)))

        self.L2 = 0
        self.H1 = 0


    def readTriangleMesh(self, filename):
        """
        Thanks to Logan Harbour
        Loads a triangle generated mesh with prefix "filename". Assumes that there
        are no attributes included in the mesh
        Args:
          filename (string): The mesh prefix
        Returns:
          nodes (np.ndarray): The nodes in the mesh, in which each component is
                              the vertices of a given node.
          edges (list of tuples): The edges in the mesh, in which each tuple in the
                                  list contains a tuple of the node ids of said
                                  edge and the edges boundary marker.
          elems (list of tuples): The elements in the mesh, in which each tuple in
                                  the list contains the node ids of each element.
        """

        # Load nodes
        data = np.loadtxt(filename + '.node', skiprows=1)
        for row in data:
            vertex = row[1:3]
            self.nodes.append(vertex)

        # Load edges
        data = np.loadtxt(filename + '.edge', skiprows=1)
        for row in data:
            endpoints = (int(row[1]) - 1, int(row[2]) - 1)
            boundary_marker = int(row[3])
            self.edges.append((endpoints, boundary_marker))

        # Load elems
        data = np.loadtxt(filename + '.ele', skiprows=1)
        for row in data:
            node_ids = (int(row[1]) - 1, int(row[2]) - 1, int(row[3]) - 1)
            self.elems.append(node_ids)

    def phi_(self, point):

        f1 = 1 - point[0] - point[1]
        f2 = point[0]
        f3 = point[1]
        return f1, f2, f3

    def getArea(self, elemID):
        node0 = self.nodes[self.elems[elemID][0]]
        node1 = self.nodes[self.elems[elemID][1]]
        node2 = self.nodes[self.elems[elemID][2]]

        avec = [node1[0]-node0[0], node1[1]-node0[1]]
        bvec = [node2[0]-node0[0], node2[1]-node0[1]]
        area = np.abs(np.cross(avec, bvec)) * 0.5
        return area

    def createElementMass(self, elemID):
        '''Needs to be called at each element'''
        elemmass = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                for si in range(self.qweights.size):
                    elemmass[i][j] += self.phi_(self.qpoints[si])[i] \
                                      * self.phi_(self.qpoints[si])[j] \
                                      * self.qweights[si]
        elemmass = self.q * elemmass * 2 * self.getArea(elemID)
        return elemmass

    def create_D(self, elemID):
        '''Creates elemenal lumped mass matrix; inversion needs to be done'''
        '''after D_e has been globalized'''
        D = np.zeros((3,3))
        area = self.getArea(elemID)
        D[0][0] = area / 3
        D[1][1] = area / 3
        D[2][2] = area / 3
        return D

    def createElementStiff(self, elemID):
        node0 = self.nodes[self.elems[elemID][0]]
        node1 = self.nodes[self.elems[elemID][1]]
        node2 = self.nodes[self.elems[elemID][2]]
        b2 = node2[1] - node0[1]
        b3 = node0[1] - node1[1]
        c2 = node0[0] - node2[0]
        c3 = node1[0] - node0[0]
        B_inv = np.array([[b2, c2],
                          [b3, c3]])
        area = self.getArea(elemID)
        elemstiff = 1/4/area * np.matmul(self.gradphi.T, np.matmul(B_inv, np.matmul(B_inv.T, self.gradphi)))
        return elemstiff

    def createElementF(self, elemID, tval):
        node0 = self.nodes[self.elems[elemID][0]]
        node1 = self.nodes[self.elems[elemID][1]]
        node2 = self.nodes[self.elems[elemID][2]]
        qp = self.qpoints
        qw = self.qweights
        elemF = np.zeros((3,1))
        for oi in range(3):
            for i in range(qw.size):
                x = (node1[0]-node0[0])*qp[i][0]+(node2[0]-node0[0])*qp[i][1]+node0[0]
                y = (node1[1]-node0[1])*qp[i][0]+(node2[1]-node0[1])*qp[i][1]+node0[1]
                if self.prob == 0:
                    f = np.exp(-tval)*((self.q+10*m.pi**2)*tval+(1-tval))*m.cos(3*m.pi*x)*m.cos(m.pi*y)
                elif self.prob == 1:
                    f = 1
                elif self.prob == 2:
                    f = x*y
                else:
                    raise ValueError('Problem Switch must be 0 thru 2.')
                elemF[oi] += 2 * self.getArea(elemID) * qw[i] * f * self.phi_(qp[i])[oi]
        return elemF

    def globalize(self, elemID, matrix):
        '''This must be called at same element as stiff, mass, and F;'''
        '''use only for elemental mass, stiffness, and F'''
        node0id = self.elems[elemID][0]
        node1id = self.elems[elemID][1]
        node2id = self.elems[elemID][2]
        n_list = [node0id, node1id, node2id]
        if np.shape(matrix)[1] == 3:
            for i in range(len(n_list)):
                for j in range(len(n_list)):
                    self.rows.append(n_list[i])
                    self.cols.append(n_list[j])
                    self.vals.append(matrix[i][j])
                    # self.g_A[n_list[i]][n_list[j]] += matrix[i][j]
        else: #np.shape(matrix)[1] == 1:
            for k in range(len(n_list)):
                self.g_F[n_list[k]] += matrix[k]
        # else:
        #     raise ValueError('Input matrix must be 3x3 or 3x1.')

    def globalizeD(self, elemID, Dmatrix):
        '''Use only for D_e'''
        node0id = self.elems[elemID][0]
        node1id = self.elems[elemID][1]
        node2id = self.elems[elemID][2]
        n_list = [node0id, node1id, node2id]
        for i in range(len(n_list)):
            self.Drows.append(n_list[i])
            self.Dcols.append(n_list[i])
            self.Dvals.append(Dmatrix[i][i])

    def applyNeumann(self):
        '''after elem loop before solve'''
        for i in range(len(self.edges)):
            if self.edges[i][1] == 1 or self.edges[i][1] == 2:
                node0id = self.edges[i][0][0]
                node1id = self.edges[i][0][1]
                length = np.sqrt((self.nodes[node1id][0]-self.nodes[node0id][0])**2
                               + (self.nodes[node1id][1]-self.nodes[node0id][1])**2)
                self.g_F[node0id] += self.g * 0.5 * length
                self.g_F[node1id] += self.g * 0.5 * length

    def getExplicitTimeStep(self):
        '''This is a multipurpose function, not good coding practice. Gets D and A'''
        '''for either of the 2 forward solving methods, but also gets time steps'''
        '''for stable forward solve'''
        s_D = csc_matrix((self.Dvals, (self.Drows, self.Dcols)), dtype=float)
        s_A = csc_matrix((self.vals, (self.rows, self.cols)), dtype=float)
        s_Dinv = inv(s_D)
        if self.scheme == 1:
            eig, vec = eigs(s_Dinv @ s_A, k=1)
            tau = 2/eig[0].real
            # print(tau)
            self.tau = tau - 0.02*tau
            self.tpnts = int(0.006 // self.tau + 2) # floor div + 1 + 1 to make it # pnts not # intervals
            print(self.tpnts)
            self.U = np.zeros((self.tpnts, len(self.nodes))) # This is real nasty
            self.tvals = np.linspace(0, 0.006, self.tpnts)
            print(self.tvals[1]-self.tvals[0])
            # self.tvals = np.linspace(0, self.tpnts*self.tau, self.tpnts)
        return s_A, s_Dinv # these are needed in the forwardSolve

    def backwardSolve(self, f_matrix, timeid):
        '''Can only be used in appropriate for loop'''
        s_D = csc_matrix((self.Dvals, (self.Drows, self.Dcols)), dtype=float)
        s_A = csc_matrix((self.vals, (self.rows, self.cols)), dtype=float)
        RHS = s_D @ self.U[timeid] + self.tau * f_matrix
        LHS = s_D + self.tau*s_A
        self.U[timeid+1] = spsolve(LHS, RHS)

    def forwardSolve(self, timeid, s_A, Dinv, f_matrix):

        self.U[timeid+1] = self.U[timeid] - self.tau * (Dinv @ (s_A @ self.U[timeid])) \
                            + self.tau * (Dinv @ f_matrix)

    def solve(self):
        '''converts g_A to sparse and sparse solves the system'''
        sparse_g_A = csr_matrix((self.vals, (self.rows, self.cols)), dtype=float)
        # sparse_g_A = csr_matrix(self.g_A, dtype=float)
        self.U = spsolve(sparse_g_A, self.g_F)

    def interpolate(self, tval):
        if tval not in self.tvals:
            for tid in range(self.tpnts-1):
                if tval < self.tvals[tid+1]:
                    tslope = (tval-self.tvals[tid])/self.tau
                    interpU = self.U[tid] + (self.U[tid+1]-self.U[tid])*tslope
                    break
        else:
            list = self.tvals.tolist()
            interpU = self.U[list.index(tval)]
        return interpU

    def getL2(self, i_U, tv): # i_U is the interpolated U vector
        qp = self.qpoints
        qw = self.qweights
        domainL2 = 0
        domainH1 = 0

        for elemID in range(len(self.elems)):
            node0 = self.nodes[self.elems[elemID][0]]
            node1 = self.nodes[self.elems[elemID][1]]
            node2 = self.nodes[self.elems[elemID][2]]
            b2 = node2[1] - node0[1]
            b3 = node0[1] - node1[1]
            c2 = node0[0] - node2[0]
            c3 = node1[0] - node0[0]
            B_inv = np.array([[b2, c2],
                              [b3, c3]])
            area = self.getArea(elemID)

            elementL2 = 0
            elementH1 = 0
            for i in range(qw.size):
                mysoln_ = 0
                mysoln_p = 0
                for oi in range(3):
                    phi_ = self.phi_(qp[i])[oi]
                    mysoln_ += i_U[self.elems[elemID][oi]]*phi_
                    mysoln_p += i_U[self.elems[elemID][oi]]*self.gradphi[:,oi]
                mysoln_p_fin = 1/2/area*np.matmul(B_inv.T, mysoln_p)

                x = (node1[0]-node0[0])*qp[i][0]+(node2[0]-node0[0])*qp[i][1]+node0[0]
                y = (node1[1]-node0[1])*qp[i][0]+(node2[1]-node0[1])*qp[i][1]+node0[1]
                if self.prob == 0:
                    true_soln_ = tv*np.exp(-tv)*m.cos(3*m.pi*x)*m.cos(m.pi*y)
                    true_soln_p = [-tv*np.exp(-tv)*3*m.pi*m.sin(3*m.pi*x)*m.cos(m.pi*y),
                                   -tv*np.exp(-tv)*m.pi*m.cos(3*m.pi*x)*m.sin(m.pi*y)]
                else:
                    return None
                # if elemID == 60 and i == 2:
                #     print(x)
                #     print(y)
                #     print(mysoln_)
                #     print(true_soln_)
                elementL2 += 2 * area * (mysoln_-true_soln_)**2 * qw[i]
                uhp_up_2 = np.dot((mysoln_p_fin-true_soln_p),(mysoln_p_fin-true_soln_p))
                # if i == 0 and elemID == 3:
                #     print(mysoln_p_fin)
                #     print(true_soln_p)
                #     print(uhp_up_2)
                elementH1 += 2 * area * (uhp_up_2
                                      + (mysoln_-true_soln_)**2) * qw[i]
                # print(elementH1)
            domainL2 += elementL2
            domainH1 += elementH1
        self.L2 = m.sqrt(domainL2)
        self.H1 = m.sqrt(domainH1)

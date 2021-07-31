import numpy as np
from PA_7_Utilities import TFEM
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def execute():
    # sd_list = [1, 2, 3]
    td_list = [20]#, 40, 80]
    P = 0 # 0, 1, or 2
    for sdi in range(3):
        for tdi in range(len(td_list)):
            start_time = time.time()
            t = TFEM('Meshes/Problem{}.{}'.format(P+1,sdi+1), td_list[tdi], P, 0)
            # Problem 1, 2, 3; .spatial_discretization, num time points, problem 0, 1, 2,
            # 0 for backward, 1 for forward stable, 2 forward unstable

            for elemID in range(len(t.elems)):
                D_e = t.create_D(elemID)
                t.globalizeD(elemID, D_e)
                A_0_e = t.createElementMass(elemID)
                t.globalize(elemID, A_0_e)
                A_1_e = t.createElementStiff(elemID)
                t.globalize(elemID, A_1_e)
                F_e = t.createElementF(elemID, 0) # Need f stuff here so prob 1 and 2 can use it
                t.globalize(elemID, F_e)
            t.applyNeumann()
            if t.prob == 0: # only problem with time-dependent f
                t.g_F = np.zeros(len(t.nodes))
                # Backward (implicit; unconditionally stable) ##################
                if t.scheme == 0:
                    for timeid in range(len(t.tvals)-1):
                        t.g_F = np.zeros(len(t.nodes))
                        for elemID in range(len(t.elems)):
                            F_e = t.createElementF(elemID, t.tvals[timeid+1])
                            t.globalize(elemID, F_e)
                        t.applyNeumann()
                        t.backwardSolve(t.g_F, timeid)
                    print(t.U[-1])
                ################################################################
                # Forward (explicit) ###########################################
                elif t.scheme == 1 or t.scheme == 2:
                    sA, Dinv = t.getExplicitTimeStep()
                    for timeid in range(len(t.tvals)-1):
                        t.g_F = np.zeros(len(t.nodes)) # Dangerous
                        for elemID in range(len(t.elems)):
                            F_e = t.createElementF(elemID, t.tvals[timeid])
                            t.globalize(elemID, F_e)
                        t.applyNeumann()
                        t.forwardSolve(timeid, sA, Dinv, t.g_F)


                ################################################################
                if tdi == 0:
                    print(len(t.elems))
                if t.scheme == 0 or t.scheme == 2:
                    tv = 1
                    interpU = t.interpolate(tv)
                    t.getL2(interpU, tv)
                    print('{:.3E}'.format(t.L2), '{:.3E}'.format(t.H1))

                    tv = 3
                    interpU = t.interpolate(tv)
                    t.getL2(interpU, tv)
                    print('{:.3E}'.format(t.L2), '{:.3E}'.format(t.H1))
                    print("%s seconds" % (time.time() - start_time))
                    print('=================================')
                elif t.scheme == 1:
                    tv = 0.002
                    # tv = t.tvals[t.tpnts // 3]
                    interpU = t.interpolate(tv)
                    t.getL2(interpU, tv)
                    print('t = {}'.format(tv))
                    print('{:.3E}'.format(t.L2), '{:.3E}'.format(t.H1))

                    tv = t.tvals[-1]
                    interpU = t.interpolate(tv)
                    t.getL2(interpU, tv)
                    print('t = {}'.format(tv))
                    print('{:.3E}'.format(t.L2), '{:.3E}'.format(t.H1))
                    print("%s seconds" % (time.time() - start_time))
                    print('=================================')
                else:
                    raise ValueError('time scheme must be 0, 1, or 2.')


# Problems 1 and 2 #############################################################
################################################################################

            elif t.prob == 1 or t.prob == 2:
                # Backward #####################################################
                if t.scheme == 0:
                    for timeid in range(len(t.tvals)-1):
                        t.backwardSolve(t.g_F, timeid)
                ################################################################
                # Forward ######################################################
                elif t.scheme == 1 or t.scheme == 2:
                    print(len(t.elems))
                    s_A, Dinv = t.getExplicitTimeStep()
                    for timeid in range(len(t.tvals)-1):
                        t.forwardSolve(timeid, s_A, Dinv, t.g_F)
                ################################################################
                if tdi == 0:
                    numelems = len(t.elems)
                if t.scheme == 0 or t.scheme == 2:
                    tv1 = 1
                    interpU1 = t.interpolate(tv1)
                    tv3 = 3
                    interpU3 = t.interpolate(tv3)
                elif t.scheme == 1:
                    tv1 = 1#t.tvals[t.tpnts // 3]
                    interpU1 = t.interpolate(tv1)
                    tv3 = t.tvals[-1]
                    interpU3 = t.interpolate(tv3)

                x = np.zeros(len(t.nodes))
                y = np.zeros(len(t.nodes))
                for i in range(len(t.nodes)):
                    x[i] = t.nodes[i][0]
                    y[i] = t.nodes[i][1]
                tri_list = tr.Triangulation(x,y,triangles=t.elems)
                # t = 1 plot ###################################################
                fig1 = plt.figure()
                ax1 = fig1.gca(projection='3d')
                phlot1 = Axes3D.plot_trisurf(ax1, x, y, interpU1, triangles=tri_list.triangles, cmap = cm.magma)
                ax1.set_xlabel('$x$')
                ax1.set_ylabel('$y$')
                ax1.set_zlabel('$u_h$')
                ax1.set_title('{} spatial elements, {} time pnts, t={:.1E}'.format(numelems, t.tpnts, tv1))
                for spine in ax1.spines.values():
                    spine.set_visible(False)
                fig1.tight_layout()
                # fig1.colorbar(phlot1)
                if t.scheme == 0:
                    fig1.savefig('t1T{}S{}P{}Back.pdf'.format(t.tpnts,numelems,P))
                elif t.scheme == 1:
                    fig1.savefig('t1T{}S{}P{}Stable.pdf'.format(t.tpnts,numelems,P))
                elif t.scheme == 2:
                    fig1.savefig('t1T{}S{}P{}Unstable.pdf'.format(t.tpnts,numelems,P))
                # plt.show()
                ################################################################

                # t = 3 plot ###################################################
                fig2 = plt.figure()
                ax2 = fig2.gca(projection='3d')
                phlot2 = Axes3D.plot_trisurf(ax2, x, y, interpU3, triangles=tri_list.triangles, cmap = cm.magma)
                ax2.set_xlabel('$x$')
                ax2.set_ylabel('$y$')
                ax2.set_zlabel('$u_h$')
                ax2.set_title('{} spatial elements, {} time pnts, t={:.1E}'.format(numelems, t.tpnts, tv3))
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                fig2.tight_layout()
                # fig2.colorbar(phlot)
                if t.scheme == 0:
                    fig2.savefig('t3T{}S{}P{}Back.pdf'.format(t.tpnts,numelems,P))
                elif t.scheme == 1:
                    fig2.savefig('t3T{}S{}P{}Stable.pdf'.format(t.tpnts,numelems,P))
                elif t.scheme == 2:
                    fig2.savefig('t3T{}S{}P{}Unstable.pdf'.format(t.tpnts,numelems,P))
                # plt.show()

execute()

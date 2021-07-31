lamb = mpmath.lambertw(0.000679545158442411*(self.Tright + 215.0) *
                             np.exp(0.000199203187250996*self.mesh.xend**2 *
                             self.q + 0.000597609561752988*self.Tright -
                             0.000199203187250996*self.q*x**2))
      self.Ttrue.append(1673.33333333333 * lamb - 215.0)

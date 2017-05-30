# -*- coding: utf-8 -*-
# AdaptiveDEigenSolve.py
"""
Tools to find eigenvalues of a one dimensional differential operator. Adaptively
    refines a mesh using mid-point finite differences.

@version: 05.30.2017
@author: luke_wortsmann
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline


def getFiniteDifference(dx, n=1):
    # Find nth order finite difference coefficents for gridpoints dx
    a = np.zeros(len(dx))
    a[n] = np.math.factorial(n)
    A = np.array([np.power(dx, i) for i in range(len(dx))])
    return np.linalg.solve(A, a)


class FunctionSpace:

    def __init__(self, x, deg=2):
        self.x    = x
        self.deg  = deg
        self.nV   = len(x)
        self.pdeg = 2 * self.deg + 1

    def D(self, n=1):
        # Returns the nth derivative operator
        if n == 0:
            return self.I()
        data, iA, jA = np.zeros((3, self.nV, self.pdeg))
        for i in range(self.nV):
            if i - self.deg < 0:
                sten = self.x[:self.pdeg]
                jA[i] = np.arange(self.pdeg)
            elif i + self.deg + 1 > self.nV:
                sten  = self.x[-self.pdeg:]
                jA[i] = np.arange(self.nV - self.pdeg, self.nV)
            else:
                sten = self.x[i - self.deg: i + self.deg + 1]
                jA[i] = np.arange(i - self.deg, i + self.deg + 1)
            data[i] = getFiniteDifference(sten - self.x[i], n=n)
            iA[i] = i
        data, iA, jA = data.flatten(), iA.flatten(), jA.flatten()
        A = sparse.coo_matrix((data, (iA, jA)), shape=(self.nV, self.nV))
        return A.tocsc()

    def I(self):
        # Returns the identity operator
        A = sparse.identity(self.nV)
        return A.tocsc()

    def F(self, f):
        # Projects function f onto the function space
        f_eval = np.vectorize(f)
        dia = f_eval(self.x)
        A = sparse.dia_matrix(([dia], [0]), shape=(self.nV, self.nV))
        return A.tocsc()

    def BC(self, A, left=0.0, right=0.0):
        # applies boundary conditions on operator A
        if left is not None:
            A[0]   *= 0.0
            A[0, 0] = left
        if right is not None:
            A[-1]    *= 0.0
            A[-1, -1] = right
        A.eliminate_zeros()
        return A.tocsc()


def fixPhase(v0, v1, tol=1e-12):
    # for two solutions (on same gridpoints), ensure both have the same phase
    phase = lambda k: np.sum(np.abs(v0 - np.exp(1j * k) * v1))
    k = minimize_scalar(phase, bounds=[0.0, 2.0 * np.pi], tol=tol).x
    return np.real_if_close(np.exp(1j * k)) * v1


def refine(x, err, tol=1e-10):
    # refine the mesh based on error estimates
    err_p = (err[1:] + err[:-1]) / 2
    x_p, nx = [], len(x)
    for i, xi in enumerate(x):
        x_p.append(xi)
        if (i + 1 < nx) and (err_p[i] >= tol):
            mid = ((x[i + 1] - xi) / 2)
            x_p.append(xi + mid)
    return np.array(x_p)


def adaptiveSolver(L, l, u, sigma, ldeg=2, udeg=3, tol=1e-10, nx=50, max_steps=50, verbose=True):
    """
    Find the eigenvalue/eigenfunction of the differential operator L on the
    compact domain from l to u.

    Arguments:
        L:      Function on a FunctionSpace, see example code below
        l:      Lower edge of domain
        u:      Upper edge of domain
        sigma:  Find eigenvalue near sigma

    Parameters:
        ldeg:   Degree of lower function space (for adaptive error)
        udeg:   Degree of upper function space (for adaptive error)
                Finite difference degree is 2 * deg, uses 2 * deg + 1 points
        tol:    Tolerance of adaptive error scheme
        nx:     Inital number of gridpoints
        max_steps:  Maximum number of refinements
        verbose:    Prints eigenvalues and errors

    Returns:
        w:  Eigenvalue
        vr: Spline of real part of eigenfunction
        vi: Spline of imaginary part of eigenfunction
    """
    x_p       = np.linspace(l, u, nx)
    converged = False
    for n in range(max_steps):
        x, nx  = x_p, len(x_p)
        phi_u  = FunctionSpace(x, deg=udeg)
        phi_l  = FunctionSpace(x, deg=ldeg)
        Lu, Ll = L(phi_u), L(phi_l)
        wl, vl = sparse.linalg.eigs(Ll, sigma=sigma, k=1)
        wu, vu = sparse.linalg.eigs(Lu, sigma=sigma, k=1)
        vl, vu = vl[:, 0], vu[:, 0]
        vu     = fixPhase(vl, vu)
        err    = np.abs(vl - vu)
        if verbose:
            print 'Iteration %s:\t%f\t%f'%(n, np.abs(wl[0]), np.abs(wu[0]))
            print 'Mean Error: \t%f'%np.log10( np.mean(err) )
            print
        x_p = refine(x, err, tol=tol)
        if len(x_p) == nx:
            converged = True
            break
    if converged is False:
        print 'Convergence Error'
    # Normalize
    a = np.sqrt( UnivariateSpline(x, np.abs(vu)**2.0, s=0).integral(l, u) )
    vu /= a
    vr = UnivariateSpline(x, np.real(vu), s=0, ext=1)
    vi = UnivariateSpline(x, np.imag(vu), s=0, ext=1)
    return wu[0], vr, vi


if __name__ == '__main__':
    # Define an operator, should operate on a function space:

    def L(phi):
        # phi is a FunctionSpace
        # Calls to phi.D(), phi.F(), ect... return sparse matricies
        Lop  = phi.F(lambda x: x**2 - 1        ) * phi.D(2)
        Lop += phi.F(lambda x: (3*x) - (1.0/x) ) * phi.D(1)
        Lop += phi.F(lambda x: 1.0 / (x**2)    ) * phi.I()
        # specify boundary conditions (also returns sparse matrix):
        return phi.BC(Lop, right=None)

    # Call the adaptive solver (l is nonzero to prevent divide by zero issue)
    w, vr, vi = adaptiveSolver(L, l=1e-25, u=1.0, sigma=100.0, verbose=True, tol=1e-6)

    # Print the eigenvalue, should be exactly 99
    print 'Eigenvalue: ', round(np.real_if_close(w), 3)

    # Plot the eigenfunction:
    import matplotlib.pyplot as plt
    x = np.linspace(0.0, 1.0, 1000)
    plt.title('Solution')
    plt.plot(x, vr(x))
    plt.show()

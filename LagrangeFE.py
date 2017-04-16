# -*- coding: utf-8 -*-
# LagrangeFE.py
"""
Implimentation of first and second order Lagrange finite elements for the
Poisson problem:

    −grad . kappa(x, y) grad u(x, y) = f(x, x)  on region
    u(x, y) = g(x, y)                           on region boundary

An anlysis of these methods is in the accompanying IPython notebook.
For UIUC CS 555 HW 5

@version: 04.13.2017
@author: wortsma2
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla


def buildArrays(AA, IA, JA, bb, ib):
    A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
    A = A.tocsr()
    A = A.tocoo()
    ib = ib.ravel()
    b = sparse.coo_matrix( (bb.ravel(), (ib, 0 * ib)) )
    b = b.tocsr()
    b = np.array(b.todense()).ravel()
    return A, b


def getArea(v0, v2, v1):
    # area of a triangle
    ad = (v2[0] - v0[0]) * (v1[1] - v0[1])
    bc = (v1[0] - v0[0]) * (v2[1] - v0[1])
    return abs(ad - bc) / 2


def getProjection(S, F):
    # quadratic basis
    q = lambda v: np.array([1, v[0], v[1], v[0]**2, v[0] * v[1], v[1]**2])
    # Using stencil S, project function F onto basis
    A = np.array([q(v) for v in S])
    b = np.array([F(v[0], v[1]) for v in S])
    # handle tensor functions:
    outshape = A.shape[1:] + b.shape[1:]
    bN = b.reshape(( b.shape[0], np.prod(b.shape[1:]) ))
    # solve using lstsqs
    coef = np.linalg.lstsq(A, bN)[0]
    # get back correct shape
    return coef.reshape(outshape)


def quadBasis(v0, v1, v2):
    # Integral of a quadratic basis over triangular element
    qx, qy = (v0 + v1 + v2) / 3
    qxx, qyy = (v0**2 + v1**2 + v2**2 + v1 * v2 + v0 * (v1 + v2)) / 6
    # integral of cross term is most complicated
    qxy = 2 * v0[0] * v0[1] + 2 * v1[0] * v1[1] + 2 * v2[0] * v2[1]
    qxy += v0[1] * v1[0] + v0[0] * v1[1]
    qxy += v0[1] * v2[0] + v0[0] * v2[1]
    qxy += v1[1] * v2[0] + v1[0] * v2[1]
    qxy /= 12.
    # mulitply by area of element
    a = getArea(v0, v1, v2)
    return a * np.array([1, qx, qy, qxx, qxy, qyy])


def quadD2S7(fL, fR, vi):
    # integrate the two functions fL and fR over the triangular region
    # defined by the verticies in vi
    v0, v1, v2 = vi[:3]
    # 7 point stencil:
    vc  = (v0 + v1 + v2) / 3
    vc0 = (v0 + v1 + vc) / 3
    vc1 = (v0 + v2 + vc) / 3
    vc2 = (v1 + v2 + vc) / 3
    sV = np.array([v0, v1, v2, vc, vc0, vc1, vc2])
    # build quadrature of a quadratic basis
    quadbasis = quadBasis(v0, v1, v2)
    # project functions onto basis using stencil:
    coefL = getProjection(sV, fL)
    coefR = getProjection(sV, fR)
    quadL = np.einsum('i...,i', coefL, quadbasis)
    quadR = np.einsum('i...,i', coefR, quadbasis)
    return quadL, quadR


def quadD1S1(fL, fR, vi):
    # integrate the two functions fL and fR over the triangular region
    # defined by the verticies in vi
    v0, v1, v2 = vi[:3]
    vc  = (v0 + v1 + v2) / 3
    a = getArea(v0, v1, v2)
    # one point quadrature
    return a * fL(vc[0], vc[1]), a * fR(vc[0], vc[1])


class LagrangeFE:

    def __init__(self, V, E, f, kappa):
        self.f, self.kappa = f, kappa
        # initialize mesh
        self._init_mesh(V, E)
        # initialize basis fuctions
        self._init_basis()
        # use a 7-point quadratic stencil as the default integrator
        self.integrator = quadD2S7

    def assembleElement(self, ei):
        # calculate basis functions:
        vi = self.V[ei]
        qA = np.array([ self.q(v[0], v[1]) for v in vi ])
        phi = la.solve(qA, np.eye(self.nev)).T

        # left-hand side:
        def LHS(x, y):
            dphi = (phi.dot( self.gradq(x, y) ))
            return self.kappa(x, y) * ( dphi.dot(dphi.T) )

        # right-hand side:
        def RHS(x, y):
            return self.f(x, y) * phi.dot( self.q(x, y) )

        # integrate
        Ai, bi = self.integrator(LHS, RHS, vi)
        return Ai, bi

    def implimentBoundary(self, A, b, g):
        # impliment boundary conditions
        A = A.tocoo()
        self.u0 = np.array([ g(v[0], v[1]) for v in self.V ])
        b = b - A * self.u0
        for k in range( A.nnz ):
            i, j = A.row[k], A.col[k]
            if (i in self.B) or (j in self.B):
                if i == j:
                    A.data[k] = 1.0
                else:
                    A.data[k] = 0.0
        b[self.B] = 0.0
        return A.tocsr(), b

    def assembleMatrix(self):
        # loop over mesh elements to assemble matrix
        AA, IA, JA = np.zeros((3, self.ne, self.nev**2))
        bb, ib = np.zeros((2, self.ne, self.nev))
        for i, ei in enumerate(self.E):
            Ai, bi = self.assembleElement(ei)
            indices = [ei[i] for i in range(self.nev)]
            indicesI = [self.nev * [ei[i]] for i in range(self.nev)]
            AA[i, :] = Ai.ravel()
            IA[i, :] = np.concatenate(indicesI)
            JA[i, :] = self.nev * indices
            bb[i, :] = bi.ravel()
            ib[i, :] = indices
        # build arrays using the COO format
        return buildArrays(AA, IA, JA, bb, ib)

    def solve(self, A, b):
        # solve the matrix equation
        u = sla.spsolve(A, b)
        u += self.u0
        return u


class LinearLFE(LagrangeFE):

    def _init_mesh(self, V, E):
        self.V, self.E = V, E
        self.nv, self.ne =  self.V.shape[0], self.E.shape[0]
        self.B = self.getMeshBoundary(self.E)

    def getMeshBoundary(self, E):
        S = np.sort([E, np.roll(E, 1, axis=1)], 0).T
        S = np.concatenate(S, axis=0)
        Sset, Scount = np.unique([str(tuple(si)) for si in S], return_counts=True)
        B = np.unique([ eval(s) for s in Sset[Scount==1] ])
        return B

    def _init_basis(self):
        self.nev = 3
        self.gradq = lambda x, y: np.array([[0, 1, 0], [0, 0, 1]]).T
        self.q = lambda x, y: np.array([1, x, y])


class QuadraticLFE(LagrangeFE):

    def _init_mesh(self, V, E):

        S = np.sort([E, np.roll(E, 1, axis=1)], 0).T
        Sc = np.concatenate(S, axis=0)
        Sset = np.unique([str(tuple(si)) for si in Sc])
        V2V = np.array([ eval(s) for s in Sset ])
        Vm = V[V2V].mean(1)
        Vall = np.vstack([V, Vm])
        V2Vi = len(V) + np.arange(len(V2V), dtype=int)
        Em = np.zeros((E.shape[0], 6), dtype=int)
        Em[:, :3] = E
        for i in range(3):
            for j in range(len(E)):
                Em[j, 3 + i] = V2Vi[np.all(S[i, j, :] == V2V, 1)][0]

        self.V, self.E = Vall, Em
        self.nv, self.ne =  self.V.shape[0], self.E.shape[0]
        self.B = self.getMeshBoundary(self.E)

    def getMeshBoundary(self, E):
        S = np.sort([E[:, :3], np.roll(E[:, :3], 1, axis=1), E[:, 3:]], 0).T
        S = np.concatenate(S, axis=0)
        Sset, Scount = np.unique([str(tuple(si)) for si in S], return_counts=True)
        return np.unique([ eval(s) for s in Sset[Scount==1] ])

    def _init_basis(self):
        self.nev = 6
        self.gradq = lambda x, y: np.array([[0, 1, 0, 2 * x, y, 0],
                                            [0, 0, 1, 0, x, 2 * y]]).T
        self.q = lambda x, y: np.array([1, x, y, x**2, x * y, y**2])

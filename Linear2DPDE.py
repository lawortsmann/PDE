# -*- coding: utf-8 -*-
# Linear2DPDE.py
"""
Midterm
UIUC CS 555

For the general 2D problem with a given unstructured mesh, solve the linear
homogenous PDE of the form:

    d(f)/dt + d(A.f)/dx + d(B.f)/dy = 0

with Neumann boundary conditions and inital value f_0(x, y)

@version: 03.28.2017
@author: wortsma2
"""
import numpy as np
import scipy.sparse as sparse
import vtk_writer
from sys import stdout
import os


def checkorientation(V, E):
    sgn = np.zeros((E.shape[0],))
    for i in range(E.shape[0]):
        xi = V[E[i, :], 0]
        yi = V[E[i, :], 1]
        A = np.zeros((3, 3))
        A[:, 0] = 1.0
        A[:, 1] = xi
        A[:, 2] = yi
        sgn[i] = np.linalg.det(A)
    return sgn


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    stdout.flush()


class Linear2DPDE:

    def __init__(self, A, B, V, E, boundary, verbose=True):
        self.A, self.B = np.array(A), np.array(B)
        self.V, self.E = np.array(V), np.array(E)
        self.ne = len(self.E)

        # Neumann boundary values
        self.boundary = boundary
        # Verbosity:
        self.verbose = verbose

        # Precompute the mesh
        self._check_mesh()
        self.nx, self.ny, self.h, self.x, self.y, self.dA, Enbrs = self._precompute_mesh()
        self.mapL, self.mapR, self.mapB = self._build_mesh_maps(Enbrs)
        # Build the flux functional:
        self.flux = self._build_flux()

    def _check_mesh(self):
        sgn = checkorientation(self.V, self.E)
        I = np.where(sgn < 0)[0]
        E1 = self.E[I, 1]
        E2 = self.E[I, 2]
        self.E[I, 2] = E1
        self.E[I, 1] = E2

    def _precompute_mesh(self):
        if self.verbose:
            print('Computing mesh...')
        # Arrays for mesh data:
        nx, ny, h  = np.zeros((3, 3, self.ne))
        cx, cy, dA = np.zeros((3, self.ne))
        N          = np.array([[0, -1], [1, 0]])
        # Compute neighbors:
        ID    = np.kron( np.arange(0, self.ne), np.ones((3,)) )
        G     = sparse.coo_matrix((np.ones((self.ne * 3,)), (self.E.ravel(), ID,)))
        E2E   = G.T * G
        V2V   = G * G.T
        Enbrs = -np.ones((self.ne, 3), dtype=int)
        # Compute mesh data per element:
        for i, E_i in enumerate(self.E):
            v1, v2, v3 = self.V[E_i]
            # Edges and edge lengths
            e1, e2, e3 = v1 - v2, v2 - v3, v3 - v1
            l1, l2, l3 = np.sqrt(e1.dot(e1)), np.sqrt(e2.dot(e2)), np.sqrt(e3.dot(e3))
            # Compute edge normals
            n1, n2, n3 = (N.dot(e1)) / l1, (N.dot(e2)) / l2, (N.dot(e3)) / l3
            nx[:, i] = n1[0], n2[0], n3[0]
            ny[:, i] = n1[1], n2[1], n3[1]
            # Compute area of element
            p = (l1 + l2 + l3) / 2
            dA[i] = np.sqrt(p * (p - l1) * (p - l2) * (p - l3))
            # Compute 'height' of element relative to edges
            h[:, i]  = 2 * dA[i] / l1, 2 * dA[i] / l2, 2 * dA[i] / l3
            # element centers
            cx[i], cy[i] = np.mean(self.V[E_i], axis=0)
            # Compute neighbors:
            vi = self.E[i, :]
            nbrids = np.where(E2E[i, :].data == 2)[0]
            nbrs = E2E[i, :].indices[nbrids]
            # for each nbr, find the face it goes with
            for j in nbrs:
                vj = self.E[j, :]
                if (vi[0] in vj) and (vi[1] in vj):
                    Enbrs[i, 0] = j
                if (vi[1] in vj) and (vi[2] in vj):
                    Enbrs[i, 1] = j
                if (vi[2] in vj) and (vi[0] in vj):
                    Enbrs[i, 2] = j
        return nx, ny, h, cx, cy, dA, Enbrs

    def _build_mesh_maps(self, Enbrs):
        # get mapR, or neighbor map
        mapR = Enbrs.T
        # get boundary map
        mapB = np.where(mapR == -1)
        # get mapL, or map of current element
        mapL = np.outer(np.ones(3), np.arange(0, self.ne)).astype(int)
        return mapL, mapR, mapB

    def _build_flux(self):
        if self.verbose:
            print('Diagonalizing PDE...')
        # Decouple PDE and return lambda function for the flux,
        # General form, diagonalize PDE over each element/edge
        L = np.zeros((self.ne, 3, self.A.shape[0]))
        R = np.zeros((self.ne, 3) + self.A.shape)
        for i in range(self.ne):
            for j in range(3):
                C = self.A * self.nx[j, i] + self.B * self.ny[j, i]
                lmbd, R_i = np.linalg.eig(C)
                L[i, j] = lmbd - np.abs(lmbd)
                R[i, j] = R_i
        # function to divide fD by h and return array in correct shape:
        divH = lambda fD: np.swapaxes([fD[:, :, i] / self.h for i in range(fD.shape[2])], 1, 2)
        # Where the magic happens: flux = -sum_edges R.(L * R^T).(fD / h)
        # this is done using numpy einsum
        return lambda fD: -np.sum(np.einsum('...ij,...j', R, L * np.einsum('...ji,j...', R, divH(fD))), 1)

    def _timestep(self, f, dt):
        fL = f[self.mapL]
        fR = f[self.mapR]
        fR[self.mapB] = self.boundary * fL[self.mapB]
        return f + dt * self.flux(fR - f)

    def save_mesh(self, f, i):
        fn = self.save_dir + '/solution_%05d.vtu'%i
        vtk_writer.write_basic_mesh(self.V, self.E, cdata=f, mesh_type='tri', fname=fn)

    def _setup_savedir(self, save_dir):
        # Setup output directory
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for f in os.listdir(self.save_dir):
            os.remove(self.save_dir + '/' + f)

    def run(self, f_0, T, lmbda=0.25, save=10, save_dir='output'):
        if self.verbose:
            print('Solving PDE...')
        if (save != 0):
            self._setup_savedir(save_dir)

        f = np.array([ f_0(self.x[i], self.y[i]) for i in range(self.ne) ])
        dt = lmbda * np.min(self.h)
        t, i  = 0, 0
        while t < T:
            if self.verbose:
                progress(t, T, status='')
            if (save != 0) and ((i % save) == 0):
                self.save_mesh(f, i)
            f = self._timestep(f, dt)
            t += dt
            i += 1
        if (save != 0):
            self.save_mesh(f, i + 1)
        if self.verbose:
            print()
        return f


if __name__ == "__main__":
    import mesh_neu
    import refine_mesh

    # Define the linear Euler equations:
    A = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    B = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Import mesh and define Neumann boundary values
    V, E = mesh_neu.read_neu('square.neu')
    # V, E = refine_mesh.refine2dtri(V, E)
    boundary = np.array([-1, -1, 1])

    # Initialize solver
    solver = Linear2DPDE(A, B, V, E, boundary, verbose=True)

    # Inital conditions:
    f_0 = lambda x, y: np.array([ 0, 0, np.exp(-10 * (x**2 + y**2)) ])
    # f_0 = lambda x, y: np.array([ 0, 0, np.exp( -((x-200)**2 + (y+400)**2) / 10000. ) ])

    # Run solver
    f = solver.run(f_0, 10, lmbda=0.25, save=True, save_dir='output')

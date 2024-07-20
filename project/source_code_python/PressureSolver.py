import math
from Discretization import Discretization
from Grid import Grid
from Fields import Fields
import numpy as np
from numba import jit



@jit(nopython=True)
def _laplacian_numba(U, i, j, dx, dy):
        return (U[i + 1, j] - 2 * U[i, j] + U[i - 1, j]) / (dx ** 2) + \
               (U[i, j + 1] - 2 * U[i, j] + U[i, j - 1]) / (dy ** 2)


@jit(nopython=True)
def _sor_helper_n(P, i, j, dx, dy):
    return (P[i + 1, j] + P[i - 1, j]) / (dx ** 2) + \
            (P[i, j + 1] + P[i, j - 1]) / (dy ** 2)

@jit(nopython=True)
def solve_n(P, RS, fluid_mask, coeff, omega, dx, dy):

    for i in range(1, P.shape[0] - 1):
        for j in range(1, P.shape[1] - 1):
            if fluid_mask[i, j]:
                P[i, j] = (1.0 - omega) * P[i, j] + coeff * (_sor_helper_n(P, i, j, dx, dy) - RS[i, j])

    rloc = 0.0

    for i in range(1, P.shape[0] - 1):
        for j in range(1, P.shape[1] - 1):
            if fluid_mask[i, j]:

                val = _laplacian_numba(P, i, j, dx, dy) - RS[i, j]
                rloc += (val * val)

    #res = rloc / np.sum(fluid_mask)
    #res = np.sqrt(res)

    return rloc


class SOR:
    def __init__(self, omega):
        self._omega = omega


    def solve_naive(self, field, grid):
        dx = grid.dx()
        dy = grid.dy()

        coeff = self._omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))  # = _omega * h^2 / 4.0, if dx == dy == h

        for currentCell in grid.fluid_cells():
            i = currentCell.i()
            j = currentCell.j()

            # Exclude cells for communication
            if i != 0 and i != grid.size_x() + 1 and j != 0 and j != grid.size_y() + 1:
                field.set_p(i, j, (1.0 - self._omega) * field.p(i, j) + coeff * (Discretization.sor_helper(field.p_matrix(), i, j) - field.rs(i, j)))

        rloc = 0.0
        res = 0.0


        for currentCell in grid.fluid_cells():
            i = currentCell.i()
            j = currentCell.j()

            if i != 0 and i != grid.size_x() + 1 and j != 0 and j != grid.size_y() + 1:
                val = Discretization.laplacian(field.p_matrix(), i, j) - field.rs(i, j)
                rloc += (val * val)

        #res = rloc / len(grid.fluid_cells())
        #res = np.sqrt(res)

        return rloc
    


    def solve_vectorized(self, field: Fields, grid: Grid):

        dx = grid.dx()
        dy = grid.dy()
        p_tmp = field.p_matrix()
        rs = field.rs_matrix()
        omega = self._omega

        coeff = omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))  # = _omega * h^2 / 4.0, if dx == dy == h

        # Loop over fluid cells is necessary because of incremental dependencies

        for currentCell in grid.fluid_cells():
            i = currentCell.i()
            j = currentCell.j()

            # Removed the call to Discretization.sor_helper

            p_tmp[i, j] = (1.0 - omega) * field.p(i, j) + coeff * ((p_tmp[i + 1, j] + p_tmp[i - 1, j]) / (dx ** 2) + \
               (p_tmp[i, j + 1] + p_tmp[i, j - 1]) / (dy ** 2) - rs[i, j])

        field._P = p_tmp

        # Get the fluid cell mask
        fluid_mask = grid.get_fluid_cells_mask()

        # Updated Pressure field
        p = field.p_matrix()
        rs = field.rs_matrix()

        laplacian_p = Discretization.optimized_laplacian(p)

        # Calculate the residuals for fluid cells
        residuals = laplacian_p[fluid_mask] - rs[fluid_mask]
        rloc = np.sum(residuals ** 2)
        #res = np.sqrt(rloc / np.sum(fluid_mask))

        return rloc


    def solve_numba(self, field, grid):
        dx = grid.dx()
        dy = grid.dy()

        coeff = self._omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))  # = _omega * h^2 / 4.0, if dx == dy == h

        P = field.p_matrix()
        RS = field.rs_matrix()
        fluid_mask = grid.get_fluid_cells_mask()

        return solve_n(P, RS, fluid_mask, coeff, self._omega, dx, dy)
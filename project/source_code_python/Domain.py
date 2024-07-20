from enum import Enum
import mpi4py.MPI as MPI

class Domain:
    def __init__(self):
        # Minimum x index including ghost cells
        self.iminb = -1
        # Maximum x index including ghost cells
        self.imaxb = -1

        # Minimum y index including ghost cells
        self.jminb = -1
        # Maximum y index including ghost cells
        self.jmaxb = -1

        # Cell length
        self.dx = -1.0
        # Cell height
        self.dy = -1.0

        # Number of cells in x direction
        self.size_x = -1
        # Number of cells in y direction
        self.size_y = -1

        # Number of cells in x direction, not-decomposed
        self.domain_imax = -1
        # Number of cells in y direction, not-decomposed
        self.domain_jmax = -1

        # Neighbours
        self.neighbours = [-1, -1, -1, -1]



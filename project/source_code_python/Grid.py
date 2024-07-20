import numpy as np
import mpi4py.MPI as MPI
from typing import List
from collections import deque

from Enums import LidDrivenCavity, BorderPosition, CellType
from Cell import Cell
from Domain import Domain

class Grid:
    def __init__(self, geom_name: str, domain: Domain, iProc: int, jProc: int, rank: int, size: int):
        self._domain = domain
        self._iProc = iProc
        self._jProc = jProc

        self._size = size
        self._rank = rank

        self._cells = np.empty((self._domain.size_x + 2, self._domain.size_y + 2), dtype=object)

        self._fluid_cells_mask = np.zeros((self._domain.size_x + 2, self._domain.size_y + 2), dtype=bool)

        self._fluid_cells = deque()
        self._fixed_wall_cells = deque()
        self._fixed_wall_heat_cells = deque()
        self._fixed_wall_cold_cells = deque()
        self._moving_wall_cells = deque()
        self._inflow_wall_cells = deque()
        self._outflow_wall_cells = deque()

        self._useTemp = False
        self._domain = domain

        self._xCoord = 0
        self._yCoord = 0

        if geom_name != "NONE":
            geometry_data = np.zeros((self._domain.size_x + 2, self._domain.size_y + 2), dtype=int)

            self.parse_geometry_file(geom_name, geometry_data)

            self.assign_cell_types(geometry_data)
        else:
            self.build_lid_driven_cavity()


    def build_lid_driven_cavity(self):
        geometry_data = np.zeros((self._domain.size_x + 2, self._domain.size_y + 2), dtype=int)

        for i in range(self._domain.size_x + 2):
            for j in range(self._domain.size_y + 2):
                if (i == 0 and self._domain.iminb == 0) or (i == self._domain.size_x + 1 and self._domain.imaxb == self._domain.domain_imax + 2) or (j == 0 and self._domain.jminb == 0):
                    geometry_data[i][j] = LidDrivenCavity.fixed_wall_id
                elif j == self._domain.size_y + 1 and self._domain.jmaxb == self._domain.domain_jmax + 2:
                    geometry_data[i][j] = LidDrivenCavity.moving_wall_id

        self.assign_cell_types(geometry_data)


    def assign_cell_types(self, geometry_data):
        i = 0
        j = 0
        for j_geom in range(self._domain.size_y + 2):
            i = 0
            for i_geom in range(self._domain.size_x + 2):
                forParallelization = False

                if i_geom == 0 and self._domain.neighbours[0] != -1:
                    forParallelization = True
                elif i_geom == self._domain.size_x + 1 and self._domain.neighbours[1] != -1:
                    forParallelization = True
                elif j_geom == self._domain.size_y + 1 and self._domain.neighbours[2] != -1:
                    forParallelization = True
                elif j_geom == 0 and self._domain.neighbours[3] != -1:
                    forParallelization = True

                if not forParallelization:
                    if geometry_data[i_geom][j_geom] == 0:
                        self._cells[i, j] = Cell(i, j, CellType.FLUID)
                        self._fluid_cells.append(self._cells[i, j])
                    elif geometry_data[i_geom][j_geom] == LidDrivenCavity.moving_wall_id:
                        self._cells[i, j] = Cell(i, j, CellType.MOVING_WALL, geometry_data[i_geom][j_geom])
                        self._moving_wall_cells.append(self._cells[i, j])
                    elif geometry_data[i_geom][j_geom] == 1:
                        self._cells[i, j] = Cell(i, j, CellType.INFLOW, geometry_data[i_geom][j_geom])
                        self._inflow_wall_cells.append(self._cells[i, j])
                    elif geometry_data[i_geom][j_geom] == 2:
                        self._cells[i, j] = Cell(i, j, CellType.OUTFLOW, geometry_data[i_geom][j_geom])
                        self._outflow_wall_cells.append(self._cells[i, j])
                    elif geometry_data[i_geom][j_geom] == 4:
                        self._cells[i, j] = Cell(i, j, CellType.FIXED_WALL_HEAT, geometry_data[i_geom][j_geom])
                        self._fixed_wall_heat_cells.append(self._cells[i, j])
                    elif geometry_data[i_geom][j_geom] == 5:
                        self._cells[i, j] = Cell(i, j, CellType.FIXED_WALL_COLD, geometry_data[i_geom][j_geom])
                        self._fixed_wall_cold_cells.append(self._cells[i, j])
                    else:
                        self._cells[i, j] = Cell(i, j, CellType.FIXED_WALL, geometry_data[i_geom][j_geom])
                        self._fixed_wall_cells.append(self._cells[i, j])
                else:
                    if geometry_data[i_geom][j_geom] == 0:
                        self._cells[i, j] = Cell(i, j, CellType.FLUID)
                    elif geometry_data[i_geom][j_geom] == LidDrivenCavity.moving_wall_id:
                        self._cells[i, j] = Cell(i, j, CellType.MOVING_WALL, geometry_data[i_geom][j_geom])
                    elif geometry_data[i_geom][j_geom] == 1:
                        self._cells[i, j] = Cell(i, j, CellType.INFLOW, geometry_data[i_geom][j_geom])
                    elif geometry_data[i_geom][j_geom] == 2:
                        self._cells[i, j] = Cell(i, j, CellType.OUTFLOW, geometry_data[i_geom][j_geom])
                    elif geometry_data[i_geom][j_geom] == 4:
                        self._cells[i, j] = Cell(i, j, CellType.FIXED_WALL_HEAT, geometry_data[i_geom][j_geom])
                    elif geometry_data[i_geom][j_geom] == 5:
                        self._cells[i, j] = Cell(i, j, CellType.FIXED_WALL_COLD, geometry_data[i_geom][j_geom])
                    else:
                        self._cells[i, j] = Cell(i, j, CellType.FIXED_WALL, geometry_data[i_geom][j_geom])

                i += 1

            j += 1

        i = 0
        j = 0

        self._cells[i, j].set_neighbour(self._cells[i, j + 1], BorderPosition.TOP)
        self._cells[i, j].set_neighbour(self._cells[i + 1, j], BorderPosition.RIGHT)
        if self._cells[i, j].neighbour(BorderPosition.TOP).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.TOP)
        if self._cells[i, j].neighbour(BorderPosition.RIGHT).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.RIGHT)

        i = 0
        j = self._domain.size_y + 1
        self._cells[i, j].set_neighbour(self._cells[i, j - 1], BorderPosition.BOTTOM)
        self._cells[i, j].set_neighbour(self._cells[i + 1, j], BorderPosition.RIGHT)
        if self._cells[i, j].neighbour(BorderPosition.BOTTOM).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.BOTTOM)
        if self._cells[i, j].neighbour(BorderPosition.RIGHT).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.RIGHT)

        i = self._domain.size_x + 1
        j = self._domain.size_y + 1
        self._cells[i, j].set_neighbour(self._cells[i, j - 1], BorderPosition.BOTTOM)
        self._cells[i, j].set_neighbour(self._cells[i - 1, j], BorderPosition.LEFT)
        if self._cells[i, j].neighbour(BorderPosition.BOTTOM).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.BOTTOM)
        if self._cells[i, j].neighbour(BorderPosition.LEFT).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.LEFT)

        i = self._domain.size_x + 1
        j = 0
        self._cells[i, j].set_neighbour(self._cells[i, j + 1], BorderPosition.TOP)
        self._cells[i, j].set_neighbour(self._cells[i - 1, j], BorderPosition.LEFT)
        if self._cells[i, j].neighbour(BorderPosition.TOP).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.TOP)
        if self._cells[i, j].neighbour(BorderPosition.LEFT).type() == CellType.FLUID:
            self._cells[i, j].add_border(BorderPosition.LEFT)

        j = 0
        for i in range(1, self._domain.size_x + 1):
            self._cells[i, j].set_neighbour(self._cells[i + 1, j], BorderPosition.RIGHT)
            self._cells[i, j].set_neighbour(self._cells[i - 1, j], BorderPosition.LEFT)
            self._cells[i, j].set_neighbour(self._cells[i, j + 1], BorderPosition.TOP)
            if self._cells[i, j].neighbour(BorderPosition.RIGHT).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.RIGHT)
            if self._cells[i, j].neighbour(BorderPosition.LEFT).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.LEFT)
            if self._cells[i, j].neighbour(BorderPosition.TOP).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.TOP)

        j = self._domain.size_y + 1
        for i in range(1, self._domain.size_x + 1):
            self._cells[i, j].set_neighbour(self._cells[i + 1, j], BorderPosition.RIGHT)
            self._cells[i, j].set_neighbour(self._cells[i - 1, j], BorderPosition.LEFT)
            self._cells[i, j].set_neighbour(self._cells[i, j - 1], BorderPosition.BOTTOM)
            if self._cells[i, j].neighbour(BorderPosition.RIGHT).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.RIGHT)
            if self._cells[i, j].neighbour(BorderPosition.LEFT).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.LEFT)
            if self._cells[i, j].neighbour(BorderPosition.BOTTOM).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.BOTTOM)

        i = 0
        for j in range(1, self._domain.size_y + 1):
            self._cells[i, j].set_neighbour(self._cells[i + 1, j], BorderPosition.RIGHT)
            self._cells[i, j].set_neighbour(self._cells[i, j - 1], BorderPosition.BOTTOM)
            self._cells[i, j].set_neighbour(self._cells[i, j + 1], BorderPosition.TOP)
            if self._cells[i, j].neighbour(BorderPosition.RIGHT).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.RIGHT)
            if self._cells[i, j].neighbour(BorderPosition.BOTTOM).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.BOTTOM)
            if self._cells[i, j].neighbour(BorderPosition.TOP).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.TOP)

        i = self._domain.size_x + 1
        for j in range(1, self._domain.size_y + 1):
            self._cells[i, j].set_neighbour(self._cells[i - 1, j], BorderPosition.LEFT)
            self._cells[i, j].set_neighbour(self._cells[i, j - 1], BorderPosition.BOTTOM)
            self._cells[i, j].set_neighbour(self._cells[i, j + 1], BorderPosition.TOP)
            if self._cells[i, j].neighbour(BorderPosition.LEFT).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.LEFT)
            if self._cells[i, j].neighbour(BorderPosition.BOTTOM).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.BOTTOM)
            if self._cells[i, j].neighbour(BorderPosition.TOP).type() == CellType.FLUID:
                self._cells[i, j].add_border(BorderPosition.TOP)

        for i in range(1, self._domain.size_x + 1):
            for j in range(1, self._domain.size_y + 1):
                self._cells[i, j].set_neighbour(self._cells[i + 1, j], BorderPosition.RIGHT)
                self._cells[i, j].set_neighbour(self._cells[i - 1, j], BorderPosition.LEFT)
                self._cells[i, j].set_neighbour(self._cells[i, j + 1], BorderPosition.TOP)
                self._cells[i, j].set_neighbour(self._cells[i, j - 1], BorderPosition.BOTTOM)

                if self._cells[i, j].type() != CellType.FLUID:
                    if self._cells[i, j].neighbour(BorderPosition.LEFT).type() == CellType.FLUID:
                        self._cells[i, j].add_border(BorderPosition.LEFT)
                    if self._cells[i, j].neighbour(BorderPosition.RIGHT).type() == CellType.FLUID:
                        self._cells[i, j].add_border(BorderPosition.RIGHT)
                    if self._cells[i, j].neighbour(BorderPosition.BOTTOM).type() == CellType.FLUID:
                        self._cells[i, j].add_border(BorderPosition.BOTTOM)
                    if self._cells[i, j].neighbour(BorderPosition.TOP).type() == CellType.FLUID:
                        self._cells[i, j].add_border(BorderPosition.TOP)


        # Arrays of Cells

        for elem in self._fluid_cells:
            self._fluid_cells_mask[elem.i(), elem.j()] = True


    def parse_geometry_file(self, filedoc, geometry_data):
        if self._rank == 0:
            num_cells_in_x, num_cells_in_y, depth = 0, 0, 0
            full_geometry_data = np.zeros((self._domain.domain_imax + 2, self._domain.domain_jmax + 2), dtype=int)

            with open(filedoc, 'r') as infile:
                # First line : version
                inputLine = infile.readline().strip()
                if inputLine != "P2":
                    print("First line of the PGM file should be 'P2'")

                # Second line : comment
                inputLine = infile.readline().strip()

                # Continue with a stringstream
                ss = infile.read().split()
                ss_iter = iter(ss)

                # Third line : size
                num_cells_in_x = int(next(ss_iter))
                num_cells_in_y = int(next(ss_iter))

                # Fourth line : depth
                depth = int(next(ss_iter))

                # Following lines : data (origin of x-y coordinate system in bottom-left corner)
                for y in range(num_cells_in_y - 1, -1, -1):
                    for x in range(num_cells_in_x):
                        full_geometry_data[x, y] = int(next(ss_iter))

            # Correct ?: print("num_cells_in_x " + str(num_cells_in_x) + " num_cells_in_y " + str(num_cells_in_y) + " depth " + str(depth))

            I, J = 0, 0
            imin, jmin, imax, jmax = 0, 0, 0, 0

            for i in range(1, self._size):
                I = i % self._iProc + 1
                J = i // self._iProc + 1
                imin = (I - 1) * ((num_cells_in_x - 2) // self._iProc)
                imax = I * ((num_cells_in_x - 2) // self._iProc) + 2
                jmin = (J - 1) * ((num_cells_in_y - 2) // self._jProc)
                jmax = J * ((num_cells_in_y - 2) // self._jProc) + 2

                if I == self._iProc:
                    imax = num_cells_in_x
                if J == self._jProc:
                    jmax = num_cells_in_y

                rank_geometry_data = []
                for x in range(imin, imax):
                    for y in range(jmin, jmax):
                        rank_geometry_data.append(full_geometry_data[x, y])

                MPI.COMM_WORLD.Send(np.array(rank_geometry_data, dtype=int), dest=i, tag=999999)

            for y in range(self._domain.size_y + 2):
                for x in range(self._domain.size_x + 2):
                    geometry_data[x][y] = full_geometry_data[x, y]
        else:
            
            rank_geometry_data = np.zeros((self._domain.size_x + 2) * (self._domain.size_y + 2), dtype=int)

            MPI.COMM_WORLD.Recv(rank_geometry_data, source=0, tag=999999)

            for y in range(self._domain.size_y + 2):
                for x in range(self._domain.size_x + 2):
                    geometry_data[x][y] = rank_geometry_data[x * (self._domain.size_y + 2) + y]


    def size_x(self) -> int:
        return self._domain.size_x
    
    def size_y(self) -> int:
        return self._domain.size_y

    def cell(self, i: int, j: int) -> Cell:
        return self._cells[i, j]

    def dx(self) -> float:
        return self._domain.dx
    
    def dy(self) -> float:
        return self._domain.dy

    def domain(self) -> Domain:
        return self._domain

    def getUseTemp(self) -> bool:
        return self._useTemp
    
    def setUseTemp(self, useTemp: bool):
        self._useTemp = useTemp

    def fluid_cells(self) -> List[Cell]:
        return self._fluid_cells

    def fixed_wall_cells(self) -> List[Cell]:
        return self._fixed_wall_cells

    def fixed_wall_heat_cells(self) -> List[Cell]:
        return self._fixed_wall_heat_cells

    def fixed_wall_cold_cells(self) -> List[Cell]:
        return self._fixed_wall_cold_cells

    def moving_wall_cells(self):
        return self._moving_wall_cells

    def inflow_wall_cells(self) -> List[Cell]:
        return self._inflow_wall_cells

    def outflow_wall_cells(self) -> List[Cell]:
        return self._outflow_wall_cells

    def get_fluid_cells_mask(self):
        return self._fluid_cells_mask

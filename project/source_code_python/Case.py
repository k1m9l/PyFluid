import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import vtk
from mpi4py import MPI
from collections import deque

from Boundary import *
from Enums import LidDrivenCavity
from PressureSolver import SOR
from Discretization import Discretization
from Boundary import *
from Domain import Domain
from Fields import Fields
from Grid import Grid
from Communication import Communication

class Case:
    def __init__(self, file_name, rank, size):
        """
        Parallel constructor for the Case.

        Reads input file, creates Fields, Grid, Boundary, Solver and sets
        Discretization parameters. Creates output directory.

        :param file_name: Input file name
        :param argn: Argument count
        :param args: Argument list
        :param rank: MPI rank
        :param size: MPI size
        """

        self._case_name = ""
        self._dict_name = ""
        self._geom_name = ""
        self._prefix = ""


        self._rank = rank
        self._size = size

        # Read input parameters
        MAX_LINE_LENGTH = 1024
        file = open(file_name, 'r')
        nu = 0.0      # viscosity   
        UI = 0.0      # velocity x-direction 
        VI = 0.0      # velocity y-direction 
        PI = 0.0      # pressure 
        GX = 0.0      # gravitation x-direction 
        GY = 0.0      # gravitation y-direction 
        xlength = 0.0 # length of the domain x-dir.
        ylength = 0.0 # length of the domain y-dir.
        dt = 0.0      # time step 
        imax = 0      # number of cells x-direction
        jmax = 0      # number of cells y-direction
        gamma = 0.0   # uppwind differencing factor
        omg = 0.0     # relaxation factor 
        tau = 0.0     # safety factor for time step
        itermax = 0   # max. number of iterations for pressure per time step 
        eps = 0.0     # accuracy bound for pressure
        UIN = 0.0     # inlet velocity x direction
        VIN = 0.0     # inlet velocity y direction
        POut = 0.0    # outlet pressure
        TI = 0.0      # initial temperature
        TH = 0.0      # hot wall temperature
        TC = 0.0      # cold wall temperature
        alpha = 0.0   # thermal diffusivity
        beta = 0.0    # thermal expansion

        self._useTemp = False

        if file:
            lines = file.readlines()
            for line in lines:
                split_line = line.split()
                if not split_line: # empty line
                    continue
                var = line.split()[0]
                if var[0] == '#': # ignore comment line
                    continue
                if len(split_line) < 2: # Ensure there is a value for the variable
                    continue 
                if var == "xlength":
                    xlength = float(line.split()[1])
                if var == "ylength":
                    ylength = float(line.split()[1])
                if var == "nu":
                    nu = float(line.split()[1])
                if var == "t_end":
                    self._t_end = float(line.split()[1])
                if var == "dt":
                    dt = float(line.split()[1])
                if var == "omg":
                    omg = float(line.split()[1])
                if var == "eps":
                    eps = float(line.split()[1])
                if var == "tau":
                    tau = float(line.split()[1])
                if var == "gamma":
                    gamma = float(line.split()[1])
                if var == "dt_value":
                    self._output_freq = float(line.split()[1])
                if var == "UI":
                    UI = float(line.split()[1])
                if var == "VI":
                    VI = float(line.split()[1])
                if var == "GX":
                    GX = float(line.split()[1])
                if var == "GY":
                    GY = float(line.split()[1])
                if var == "PI":
                    PI = float(line.split()[1])
                if var == "itermax":
                    itermax = int(line.split()[1])
                if var == "imax":
                    imax = int(line.split()[1])
                if var == "jmax":
                    jmax = int(line.split()[1])
                if var == "geo_file":
                    self._geom_name = line.split()[1]
                if var == "UIN":
                    UIN = float(line.split()[1])
                if var == "VIN":
                    VIN = float(line.split()[1])
                if var == "POut":
                    POut = float(line.split()[1])
                if var == "TI":
                    TI = float(line.split()[1])
                if var == "wall_temp_4":
                    TH = float(line.split()[1])
                if var == "wall_temp_5":
                    TC = float(line.split()[1])
                if var == "alpha":
                    alpha = float(line.split()[1])
                if var == "beta":
                    beta = float(line.split()[1])
                if var == "useTemp":
                    tmp = line.split()[1]
                    if tmp == "true":
                        self._useTemp = True
                if var == "iProc":
                    self._iProc = int(line.split()[1])
                    if self._iProc < 1:
                        exit(0)
                if var == "jProc":
                    self._jProc = int(line.split()[1])
                    if self._iProc < 1:
                        exit(0)
        file.close()

        if (self._iProc * self._jProc) != self._size:
            exit(0)

        wall_vel = {}
        if self._geom_name == "NONE":
            wall_vel[LidDrivenCavity.moving_wall_id] = LidDrivenCavity.wall_velocity

        self.set_file_names(file_name)

        domain = Domain()
        domain.dx = xlength / float(imax)
        domain.dy = ylength / float(jmax)
        domain.domain_imax = imax
        domain.domain_jmax = jmax


        self.build_domain(domain, imax, jmax)

        self._grid = Grid(self._geom_name, domain, self._iProc, self._jProc, self._rank, self._size)

        self._grid.setUseTemp(self._useTemp)
        self._field = Fields(self._grid, nu, dt, tau, alpha, beta, UI, VI, PI, TI, GX, GY)

        self._discretization = Discretization(domain.dx, domain.dy, gamma)
        self._pressure_solver = SOR(omg)
        self._max_iter = itermax
        self._tolerance = eps

        self._boundaries = deque()

        wall_temp_adiabatic = {3: -1}
        wall_temp_heat = {4: TH}
        wall_temp_cold = {5: TC}

        if self._grid.moving_wall_cells():
            self._boundaries.append(MovingWallBoundary(self._grid.moving_wall_cells(), LidDrivenCavity.wall_velocity))
        if self._grid.fixed_wall_cells():
            self._boundaries.append(FixedWallBoundary(self._grid.fixed_wall_cells(), wall_temp_adiabatic))
        if self._grid.fixed_wall_heat_cells():
            self._boundaries.append(FixedWallBoundary(self._grid.fixed_wall_heat_cells(), wall_temp_heat))
        if self._grid.fixed_wall_cold_cells():
            self._boundaries.append(FixedWallBoundary(self._grid.fixed_wall_cold_cells(), wall_temp_cold))
        if self._grid.inflow_wall_cells():
            self._boundaries.append(InflowBoundary(self._grid.inflow_wall_cells(), UIN, VIN))
        if self._grid.inflow_wall_cells():
            self._boundaries.append(OutflowBoundary(self._grid.outflow_wall_cells(), POut))


    

    def set_file_names(self, file_name: str):
        """
        Creating file names from given input data file

        Extracts path of the case file and creates code-readable file names
        for outputting directory and geometry file.

        :param file_name: Input data file
        """
        
        temp_dir = ""
        case_name_flag = True
        prefix_flag = False

        for i in range(len(file_name) - 1, -1, -1):
            if file_name[i] == '/':
                case_name_flag = False
                prefix_flag = True
            if case_name_flag:
                self._case_name += file_name[i]
            if prefix_flag:
                self._prefix += file_name[i]

        for i in range(len(file_name) - len(self._case_name) - 1, -1, -1):
            temp_dir += file_name[i]

        self._case_name = self._case_name[::-1]
        self._prefix = self._prefix[::-1]
        temp_dir = temp_dir[::-1]

        self._case_name = self._case_name[:-4]
        self._dict_name = temp_dir
        self._dict_name += self._case_name
        self._dict_name += "_Output"

        if self._geom_name != "NONE":
            self._geom_name = self._prefix + self._geom_name

        folder = vtk.vtkDirectory()

        try:
            os.makedirs(self._dict_name, exist_ok=True)
            print(f"Directory {self._dict_name} was created successfully.\n")
        except PermissionError:
            print(f"Permission denies: Unable to create directory {self._dict_name}.")

        folder.Open(self._dict_name)
        if folder.Open(self._dict_name) == 0:
            print("Output directory could not be created.")
            print("Make sure that you have write permissions to the corresponding location")


    def simulate(self):
        """
        Main function to simulate the flow until the end time.

        Calculates the fluxes
        Calculates the right hand side
        Solves pressure
        Calculates velocities
        Outputs the solution files
        """

       
        t = 0.0
        dt = self._field.dt()
        timestep = 0
        output_counter = 0.0
        avg_iter = 0.0
        not_conv = 0
        current_iteration = 0

        num_fluid_cells = 0

        self.output_vtk(timestep)
        timestep += 1

        while t < self._t_end:
            # print("current_iteration:", current_iteration, flush=True)

            for j in self._boundaries:
                j.applyVelocity(self._field)
                if self._useTemp:
                    j.applyTemperature(self._field)

            if self._useTemp:
                self._field.calculate_temperatures_numba(self._grid) # numba
                Communication.communicate(self._field.t_matrix(), self._grid.domain(), self._rank)

            self._field.calculate_fluxes_numba(self._grid) # numba

            for j in self._boundaries:
                j.optimized_applyFlux(self._field)

            Communication.communicate(self._field.f_matrix(), self._grid.domain(), self._rank)
            Communication.communicate(self._field.g_matrix(), self._grid.domain(), self._rank)

            self._field.calculate_rs_numba(self._grid) # numba

            it = 0
            res = 1000.0

            while it < self._max_iter and res >= self._tolerance:
                
                for j in self._boundaries:
                    j.applyPressure(self._field)


                res = self._pressure_solver.solve_numba(self._field, self._grid) # numba
                it += 1

                res = Communication.reduce_sum(res)

                num_fluid_cells = len(self._grid.fluid_cells())
                num_fluid_cells = Communication.reduce_sum(num_fluid_cells)

                res = res / num_fluid_cells
                res = np.sqrt(res) 

                Communication.communicate(self._field.p_matrix(), self._grid.domain(), self._rank)

            if res >= self._tolerance:
                not_conv += 1

            avg_iter += it

    
            self._field.calculate_velocities_numba(self._grid) # numba

            Communication.communicate(self._field.u_matrix(), self._grid.domain(), self._rank)
            Communication.communicate(self._field.v_matrix(), self._grid.domain(), self._rank)

            output_counter += dt

            if output_counter >= self._output_freq:
                self.output_vtk(timestep)
                timestep += 1
                output_counter = 0

            if current_iteration % 100 == 0:
                log_message = f"iteration: {current_iteration} | t: {t:.3f} | step-size: {dt:.3f}"
                print(log_message, flush=True)

            t += dt
            current_iteration += 1

            dt = self._field.calculate_dt_naive(self._grid) # vectorized
            dt = Communication.reduce_min(dt)

        if self._rank == 0:
            print(f"\nSimulation finished\n")
            avg_iter /= current_iteration
            summary_message = f"av_iterations: {avg_iter}, number not converged: {not_conv}" 
            print(summary_message, flush=True)

        self.output_vtk(timestep)
    

    def output_vtk(self, timestep: int):
        """
        Solution file outputter

        Outputs the solution files in .vtk format. Ghost cells are excluded.
        Pressure is cell variable while velocity is point variable while being
        interpolated to the cell faces

        :param t: Timestep of the solution
        """
        structuredGrid = vtk.vtkStructuredGrid()

        points = vtk.vtkPoints()

        dx = self._grid.dx()
        dy = self._grid.dy()

        x = self._grid.domain().iminb * dx
        y = self._grid.domain().jminb * dy

        y += dy
        x += dx

        z = 0
        for col in range(self._grid.domain().size_y + 1):
            x = self._grid.domain().iminb * dx
            x += dx
            for row in range(self._grid.domain().size_x + 1):
                points.InsertNextPoint(x, y, z)
                x += dx
            y += dy

        structuredGrid.SetDimensions(self._grid.domain().size_x + 1, self._grid.domain().size_y + 1, 1)
        structuredGrid.SetPoints(points)

        fixed_wall_cells = []
        for i in range(1, self._grid.size_x() + 1):
            for j in range(1, self._grid.size_y() + 1):
                if self._grid.cell(i, j).wall_id() != 0:
                    fixed_wall_cells.append(i - 1 + (j - 1) * self._grid.size_x())

        for t in range(len(fixed_wall_cells)):
            structuredGrid.BlankCell(fixed_wall_cells[t])

        Pressure = vtk.vtkDoubleArray()
        Pressure.SetName("pressure")
        Pressure.SetNumberOfComponents(1)

        Velocity = vtk.vtkDoubleArray()
        Velocity.SetName("velocity")
        Velocity.SetNumberOfComponents(3)

        Temperature = vtk.vtkDoubleArray()
        Temperature.SetName("temperature")
        Temperature.SetNumberOfComponents(1)

        vel = [0.0, 0.0, 0.0]

        for j in range(1, self._grid.domain().size_y + 1):
            for i in range(1, self._grid.domain().size_x + 1):
                pressure = self._field.p(i, j)
                Pressure.InsertNextTuple([pressure])
                vel[0] = (self._field.u(i - 1, j) + self._field.u(i, j)) * 0.5
                vel[1] = (self._field.v(i, j - 1) + self._field.v(i, j)) * 0.5
                Velocity.InsertNextTuple(vel)
                if self._useTemp:
                    temperature = self._field.t(i, j)
                    Temperature.InsertNextTuple([temperature])

        VelocityPoints = vtk.vtkDoubleArray()
        VelocityPoints.SetName("velocity")
        VelocityPoints.SetNumberOfComponents(3)

        for j in range(self._grid.domain().size_y + 1):
            for i in range(self._grid.domain().size_x + 1):
                vel[0] = (self._field.u(i, j) + self._field.u(i, j + 1)) * 0.5
                vel[1] = (self._field.v(i, j) + self._field.v(i + 1, j)) * 0.5
                VelocityPoints.InsertNextTuple(vel)

        structuredGrid.GetCellData().AddArray(Pressure)
        structuredGrid.GetCellData().AddArray(Velocity)
        structuredGrid.GetPointData().AddArray(VelocityPoints)

        if self._useTemp:
            structuredGrid.GetCellData().AddArray(Temperature)

        writer = vtk.vtkStructuredGridWriter()
        outputname = self._dict_name + '/' + self._case_name + "_" + str(self._rank) + "." + str(timestep) + ".vtk"
        writer.SetFileName(outputname)
        writer.SetInputData(structuredGrid)
        writer.Write()



    def build_domain(self, domain: Domain, imax_domain: int, jmax_domain: int):
        """
        Fill out domain object

        Fills the Domain object with the geometrical information about the domain size. In case of running the code in
        parallel, the information should correspond to the subdomain belonging to the executing MPI-process after
        decomposition.

        :param domain: Reference to the domain object
        :param imax_domain: Number of cells in x-direction for this MPI rank
        :param jmax_domain: Number of cells in y-direction for this MPI rank
        """
        comm = MPI.COMM_WORLD

        if self._rank == 0:
            for i in range(1, self._size):
                I = i % self._iProc + 1
                J = i // self._iProc + 1
                imin = (I - 1) * (imax_domain // self._iProc)
                imax = I * (imax_domain // self._iProc) + 2
                jmin = (J - 1) * (jmax_domain // self._jProc)
                jmax = J * (jmax_domain // self._jProc) + 2
                size_x = imax_domain // self._iProc
                size_y = jmax_domain // self._jProc

                if I == self._iProc:
                    imax = imax_domain + 2
                    size_x = imax - imin - 2

                if J == self._jProc:
                    jmax = jmax_domain + 2
                    size_y = jmax - jmin - 2

                neighbours = [-1, -1, -1, -1]
                if I > 1:
                    neighbours[0] = i - 1

                if J > 1:
                    neighbours[3] = i - self._iProc

                if self._iProc > 1 and I < (self._size - 1) % self._iProc + 1:
                    neighbours[1] = i + 1

                if self._jProc > 1 and J < (self._size - 1) // self._iProc + 1:
                    neighbours[2] = i + self._iProc

                comm.send(imin, dest=i, tag=999)
                comm.send(imax, dest=i, tag=998)
                comm.send(jmin, dest=i, tag=997)
                comm.send(jmax, dest=i, tag=996)
                comm.send(size_x, dest=i, tag=995)
                comm.send(size_y, dest=i, tag=994)
                comm.Send(np.array(neighbours, dtype='i'), dest=i, tag=993)

            I = self._rank % self._iProc + 1
            J = self._rank // self._iProc + 1

            domain.iminb = (I - 1) * (imax_domain // self._iProc)
            domain.imaxb = I * (imax_domain // self._iProc) + 2
            domain.jminb = (J - 1) * (jmax_domain // self._jProc)
            domain.jmaxb = J * (jmax_domain // self._jProc) + 2
            domain.size_x = imax_domain // self._iProc
            domain.size_y = jmax_domain // self._jProc
            domain.neighbours[0] = -1 # left
            domain.neighbours[1] = -1 # right
            if self._iProc > 1:
                domain.neighbours[1] = 1
            domain.neighbours[2] = -1 # top
            if self._jProc > 1:
                domain.neighbours[2] = self._iProc
            domain.neighbours[3] = -1 # bottom
        else:
            domain.iminb = comm.recv(source=0, tag=999)
            domain.imaxb = comm.recv(source=0, tag=998)
            domain.jminb = comm.recv(source=0, tag=997)
            domain.jmaxb = comm.recv(source=0, tag=996)
            domain.size_x = comm.recv(source=0, tag=995)
            domain.size_y = comm.recv(source=0, tag=994)
            neighbours = np.empty(4, dtype='i')
            comm.Recv(neighbours, source=0, tag=993)
            domain.neighbours = neighbours.tolist()

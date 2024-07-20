
import numpy as np
from numba import jit
from Grid import Grid
from Discretization import Discretization



@jit(nopython=True)
def _laplacian_numba(U, i, j, dx, dy):
        return (U[i + 1, j] - 2 * U[i, j] + U[i - 1, j]) / (dx ** 2) + \
               (U[i, j + 1] - 2 * U[i, j] + U[i, j - 1]) / (dy ** 2)


@jit(nopython=True)
def _convection_t_numba(U, V, T, i, j, dx, dy, gamma):
    duTdx = U[i, j] * (T[i, j] + T[i + 1, j]) - U[i - 1, j] * (T[i - 1, j] + T[i, j])
    duTdx += gamma * (np.abs(U[i, j]) * (T[i, j] - T[i + 1, j]) - np.abs(U[i - 1, j]) * (T[i - 1, j] - T[i, j]))
    duTdx *= (0.5 / dx)

    dvTdy = V[i, j] * (T[i, j] + T[i, j + 1]) - V[i, j - 1] * (T[i, j - 1] + T[i, j])
    dvTdy += gamma * (np.abs(V[i, j]) * (T[i, j] - T[i, j + 1]) - np.abs(V[i, j - 1]) * (T[i, j - 1] - T[i, j]))
    dvTdy *= (0.5 / dy)

    return duTdx + dvTdy

@jit(nopython=True)
def _convection_u_numba(U, V, i, j, dx, dy, gamma):
    du2dx = (0.25 / dx) * (((U[i, j] + U[i + 1, j]) * (U[i, j] + U[i + 1, j])) - (U[i - 1, j] + U[i, j]) * (U[i - 1, j] + U[i, j]))
    du2dx += (gamma * 0.25 / dx) * (np.abs(U[i, j] + U[i + 1, j]) * (U[i, j] - U[i + 1, j]) - np.abs(U[i - 1, j] + U[i, j]) * (U[i - 1, j] - U[i, j]))

    duvdy = (0.25 / dy) * ((V[i, j] + V[i + 1, j]) * (U[i, j] + U[i, j + 1]) - (V[i, j - 1] + V[i + 1, j - 1]) * (U[i, j - 1] + U[i, j]))
    duvdy += (gamma * 0.25 / dy) * (np.abs(V[i, j] + V[i + 1, j]) * (U[i, j] - U[i, j + 1]) - np.abs(V[i, j - 1] + V[i + 1, j - 1]) * (U[i, j - 1] - U[i, j]))

    return du2dx + duvdy

@jit(nopython=True)
def _convection_v_numba(U, V, i, j, dx, dy, gamma):
    dv2dy = (0.25 / dy) * ((V[i, j] + V[i, j + 1]) * (V[i, j] + V[i, j + 1]) - (V[i, j - 1] + V[i, j]) * (V[i, j - 1] + V[i, j]))
    dv2dy += (gamma * 0.25 / dy) * (np.abs(V[i, j] + V[i, j + 1]) * (V[i, j] - V[i, j + 1]) - np.abs(V[i, j - 1] + V[i, j]) * (V[i, j - 1] - V[i, j]))

    duvdx = (0.25 / dx) * ((V[i, j] + V[i + 1, j]) * (U[i, j] + U[i, j + 1]) - (V[i - 1, j] + V[i, j]) * (U[i - 1, j] + U[i - 1, j + 1]))
    duvdx += (gamma * 0.25 / dx) * (np.abs(U[i, j] + U[i, j + 1]) * (V[i, j] - V[i + 1, j]) - np.abs(U[i - 1, j] + U[i - 1, j + 1]) * (V[i - 1, j] - V[i, j]))

    return duvdx + dv2dy

@jit(nopython=True)
def _calculate_velocities_numba(P, F, G, U, V, dt, dx, dy, fluid_mask):
    dtdx = dt / dx
    dtdy = dt / dy
    for i in range(P.shape[0] - 1):
        for j in range(P.shape[1] - 1):
            if fluid_mask[i, j]:
                U[i, j] = F[i, j] - dtdx * (P[i + 1, j] - P[i, j])
                V[i, j] = G[i, j] - dtdy * (P[i, j + 1] - P[i, j])
    return U, V

@jit(nopython=True)
def _calculate_fluxes_numba(U, V, F, G, T, dt, nu, gx, gy, alpha, beta, fluid_mask, dx, dy, gamma, use_temp):
    for i in range(1, U.shape[0] - 1):
        for j in range(1, U.shape[1] - 1):
            if fluid_mask[i, j]:
                a = _convection_u_numba(U, V, i, j, dx, dy, gamma)
                F[i, j] = U[i, j] + dt * ((nu * _laplacian_numba(U, i, j, dx, dy)) - a + gx)
                b = _convection_v_numba(U, V, i, j, dx, dy, gamma)
                G[i, j] = V[i, j] + dt * ((nu * _laplacian_numba(V, i, j, dx, dy)) - b + gy)

                if use_temp:
                    F[i, j] -= gx * dt
                    G[i, j] -= gy * dt
                    F[i, j] -= beta * (dt / 2) * (T[i, j] + T[i + 1, j]) * gx
                    G[i, j] -= beta * (dt / 2) * (T[i, j] + T[i, j + 1]) * gy
    return F, G

@jit(nopython=True)
def _calculate_rs_numba(F, G, RS, dt, fluid_mask, dx, dy):
    for i in range(1, F.shape[0] - 1):
        for j in range(1, F.shape[1] - 1):
            if fluid_mask[i, j]:
                RS[i, j] = 1 / dt * ((F[i, j] - F[i - 1, j]) / dx + (G[i, j] - G[i, j - 1]) / dy)
    return RS

@jit(nopython=True)
def _calculate_temperatures_numba(U, V, T, Ttmp, dt, alpha, fluid_mask, dx, dy, gamma):
    for i in range(1, T.shape[0] - 1):
        for j in range(1, T.shape[1] - 1):
            if fluid_mask[i, j]:
                Ttmp[i, j] = T[i, j] + dt * (
                        alpha * _laplacian_numba(T, i, j, dx, dy)
                        - _convection_t_numba(U, V, T, i, j, dx, dy, gamma)
                )
    return Ttmp

""" 
@jit(nopython=True)
def _calculate_dt_numba(dx, dy, tau, dt, U, V, nu, alpha, useTemp):

    # Compute umax and vmax using vectorized operations
        umax = np.max(np.abs(U))
        vmax = np.max(np.abs(V))

        # Compute arguments
        arg1 = (1 / (2 * nu)) * (1 / ((1 / (dx ** 2)) + (1 / (dy ** 2))))
        arg2 = dx / umax
        arg3 = dy / vmax  


        #print(f"arg1: {arg1}, arg2: {arg2}, arg3: {arg3}")

        min1 = min(arg2, arg3)
        min2 = min(arg1, min1)
        # Calculate minimum timestep

        # Adjust for temperature if applicable
        if useTemp:
            arg4 = (1 / (2 * alpha)) * (1 / ((1 / (dx ** 2)) + (1 / (dy ** 2))))
            dt = min(dt, arg4)

        dt *= tau

        # Print a warning if dt is too small (optional)
        #if dt < 0.0005:
        #    print(f"dt may get really small : {dt}")
        #    print(f"vmax: {vmax}, umax: {umax}")

        return dt

 """

class Fields:
    def __init__(self, grid: Grid, _nu, _dt, _tau, alpha, beta, UI, VI, PI, TI, GX, GY):
        
        self._U = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        self._V = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        self._P = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        self._T = np.zeros((grid.size_x() + 2, grid.size_y() + 2))

        self._F = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        self._G = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        self._RS = np.zeros((grid.size_x() + 2, grid.size_y() + 2))

        self._nu = _nu
        self._gx = GX
        self._gy = GY
        self._dt = _dt
        self._tau = _tau
        self._alpha = alpha
        self._beta = beta

        fluid_mask = grid.get_fluid_cells_mask()

        self._U[fluid_mask] = UI
        self._V[fluid_mask] = VI
        self._P[fluid_mask] = PI
        self._T[fluid_mask] = TI

        """ 
        for elem in grid.fluid_cells():
            i = elem.i()
            j = elem.j()

            self._U[i, j] = UI
            self._V[i, j] = VI
            self._P[i, j] = PI
            self._T[i, j] = TI

        """

    
    
    



















    def calculate_fluxes_naive(self, grid: Grid):
        # Implementation of flux calculation
        for elem in grid.fluid_cells():
            i = elem.i()
            j = elem.j()
            if i != 0 and i != grid.size_x() + 1 and j != grid.size_x() + 1:
                a = Discretization.convection_u(self._U, self._V, i, j)
                self._F[i, j] = self._U[i, j] + self._dt * ((self._nu * Discretization.laplacian(self._U, i, j)) -
                                                            Discretization.convection_u(self._U, self._V, i, j) + self._gx)
                b = Discretization.convection_v(self._U, self._V, i, j)
                self._G[i, j] = self._V[i, j] + self._dt * ((self._nu * Discretization.laplacian(self._V, i, j)) -
                                                            Discretization.convection_v(self._U, self._V, i, j) + self._gy)

                if grid.getUseTemp():
                    # if the temperature calculation is used, we need to subtract the gx/gy terms again
                    #print(self._F[i, j], self._G[i, j], self._gx, self._dt)
                    self.set_f(i, j, self._F[i, j] - self._gx * self._dt)
                    self.set_g(i, j, self._G[i, j] - self._gy * self._dt)

                    # add the boussinesq approximation to F and G
                    #print(self._F[i, j], self._G[i, j], self._T[i, j], self._T[i + 1, j], self._T[i, j + 1])
                    self.set_f(i, j, self._F[i, j] - self._beta * (self._dt / 2) * (self._T[i, j] + self._T[i + 1, j]) * self._gx)
                    self.set_g(i, j, self._G[i, j] - self._beta * (self._dt / 2) * (self._T[i, j] + self._T[i, j + 1]) * self._gy)

    def calculate_fluxes_vectorized(self, grid: Grid):
        U, V = self._U, self._V
        dt, nu = self._dt, self._nu
        gx, gy = self._gx, self._gy
        T = self._T if grid.getUseTemp() else None
        beta = self._beta if grid.getUseTemp() else 0

        F = np.copy(U)
        G = np.copy(V)

        # Create a mask for fluid cells
        fluid_mask = grid.get_fluid_cells_mask()

        conv_u = Discretization.optimized_convection_u(U, V, grid)
        conv_v = Discretization.optimized_convection_v(U, V, grid)

        laplacian_U = Discretization.optimized_laplacian(U)
        laplacian_V = Discretization.optimized_laplacian(V)

        F[fluid_mask] = U[fluid_mask] + dt * (nu * laplacian_U[fluid_mask] - conv_u[fluid_mask] + gx)
        G[fluid_mask] = V[fluid_mask] + dt * (nu * laplacian_V[fluid_mask] - conv_v[fluid_mask] + gy)

        if grid.getUseTemp():
            F[fluid_mask] -= gx * dt
            G[fluid_mask] -= gy * dt

            T_shifted_right = np.roll(T, -1, axis=0)
            T_shifted_up = np.roll(T, -1, axis=1)

            F[fluid_mask] -= beta * (dt / 2) * (T[fluid_mask] + T_shifted_right[fluid_mask]) * gx
            G[fluid_mask] -= beta * (dt / 2) * (T[fluid_mask] + T_shifted_up[fluid_mask]) * gy


        self._F = F
        self._G = G


    def optimized_calculate_fluxes_alternativeIndices(self, grid: Grid):
        # Create arrays of the same shape as the grid
        U, V = self._U, self._V
        dt, nu = self._dt, self._nu
        gx, gy = self._gx, self._gy
        T = self._T if grid.getUseTemp() else None
        beta = self._beta if grid.getUseTemp() else 0

        # Prepare arrays for fluxes
        F = np.copy(U)
        G = np.copy(V)

        # Get the indices of the fluid cells
        fluid_cells = grid.fluid_cells()
        indices = np.array([(elem.i(), elem.j()) for elem in fluid_cells])
        i_indices, j_indices = indices[:, 0], indices[:, 1]

        conv_u = Discretization.optimized_convection_u(U, V, grid)
        conv_v = Discretization.optimized_convection_v(U, V, grid)
        
        laplacian_U = np.zeros_like(U)
        laplacian_V = np.zeros_like(V)
        laplacian_U = Discretization.optimized_laplacian(U)
        laplacian_V = Discretization.optimized_laplacian(V)

        F[i_indices, j_indices] = U[i_indices, j_indices] + dt * (nu * laplacian_U[i_indices, j_indices] - conv_u[i_indices, j_indices] + gx)
        G[i_indices, j_indices] = V[i_indices, j_indices] + dt * (nu * laplacian_V[i_indices, j_indices] - conv_v[i_indices, j_indices] + gy)

        if grid.getUseTemp():
            F[i_indices, j_indices] -= gx * dt
            G[i_indices, j_indices] -= gy * dt
            
            F[i_indices, j_indices] -= beta * (dt / 2) * (T[i_indices, j_indices] + T[i_indices + 1, j_indices]) * gx
            G[i_indices, j_indices] -= beta * (dt / 2) * (T[i_indices, j_indices] + T[i_indices, j_indices + 1]) * gy

        self._F = F
        self._G = G

    
    def calculate_fluxes_numba(self, grid: Grid):
        self._F, self._G = _calculate_fluxes_numba(self._U, self._V, self._F, self._G, self._T, self._dt, self._nu, self._gx, self._gy, self._alpha, self._beta, grid.get_fluid_cells_mask(), grid.dx(), grid.dy(), Discretization._gamma, grid.getUseTemp())


    def calculate_rs_naive(self, grid: Grid):
        # Implementation of right hand side calculation
        for elem in grid.fluid_cells():
            i = elem.i()
            j = elem.j()

            if i != 0 and j != 0 and i != grid.size_x() + 1 and j != grid.size_y() + 1:  # exclude the buffer cells
                # calculate right hand side of PPE with F and G
                self.set_rs(i, j, 1 / self._dt * ((self._F[i, j] - self._F[i - 1, j]) / grid.dx() + (self._G[i, j] - self._G[i, j - 1]) / grid.dy()))
                            

    def calculate_rs_vectorized(self, grid: Grid):

        fluid_mask = grid.get_fluid_cells_mask()

        # Calculate the difference using np.roll
        dF_dx = (self._F - np.roll(self._F, shift=1, axis=0)) / grid.dx()
        dG_dy = (self._G - np.roll(self._G, shift=1, axis=1)) / grid.dy()

        self._RS = 1 / self._dt * (dF_dx + dG_dy)


    def calculate_rs_numba(self, grid: Grid):
        self._RS = _calculate_rs_numba(self._F, self._G, self._RS, self._dt, grid.get_fluid_cells_mask(), grid.dx(), grid.dy())

        

    def calculate_velocities_naive(self, grid: Grid):
        # Implementation of velocity calculation
        for elem in grid.fluid_cells():
            i = elem.i()
            j = elem.j()

            # update u
            self.set_u(i, j, self._F[i, j] - (self._dt / grid.dx()) * (self._P[i + 1, j] - self._P[i, j]))

            # update v
            self.set_v(i, j, self._G[i, j] - (self._dt / grid.dy()) * (self._P[i, j + 1] - self._P[i, j]))


    def calculate_velocities_vectorized(self, grid: Grid):
        P = self._P
        F = self._F
        G = self._G
        dt = self._dt
        dx = grid.dx()
        dy = grid.dy()

        P_shifted_right = np.roll(P, -1, axis=0)
        P_shifted_up = np.roll(P, -1, axis=1)

        # Create arrays of the same shape as the grid
        U = np.copy(self._U)
        V = np.copy(self._V)

        fluid_mask = grid.get_fluid_cells_mask()

        U[fluid_mask] = F[fluid_mask] - (dt / dx) * (P_shifted_right[fluid_mask] - P[fluid_mask])
        V[fluid_mask] = G[fluid_mask] - (dt / dy) * (P_shifted_up[fluid_mask] - P[fluid_mask])

        self._U = U
        self._V = V


    def calculate_velocities_numba(self, grid):
        P = self._P
        F = self._F
        G = self._G
        dt = self._dt
        dx = grid.dx()
        dy = grid.dy()
        U = np.copy(self._U)
        V = np.copy(self._V)
        fluid_mask = grid.get_fluid_cells_mask()

        U, V = _calculate_velocities_numba(P, F, G, U, V, dt, dx, dy, fluid_mask)

        self._U = U
        self._V = V


    def calculate_temperatures_naive(self, grid):
        # Implementation of temperature calculation
        # create new variable _Ttmp to store intermediate results, as e.g. in the convective term, we use T(i-1,j) to
        # calculate T(i,j) and we do not like to use the updated value already
        _Ttmp = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        
        for elem in grid.fluid_cells():
            i = elem.i()
            j = elem.j()
            if i != 0 and j != 0 and i != grid.size_x() + 1 and j != grid.size_y() + 1:  # exclude the buffer cells
                _Ttmp[i, j] = self._T[i, j] + self._dt * (
                        self._alpha * Discretization.laplacian(self._T, i, j)
                        - Discretization.convection_t(self._U, self._V, self._T, i, j)
                )
        # call copy constructor
        self._T = _Ttmp
        

    def calculate_temperatures_vectorized(self, grid):

        fluid_mask = grid.get_fluid_cells_mask()

        # Apply calculations only to interior cells
        _Ttmp = np.zeros_like(self._T)
        _Ttmp[fluid_mask] = self._T[fluid_mask] + self._dt * (
            self._alpha * Discretization.optimized_laplacian(self._T)[fluid_mask] -
            Discretization.optimized_convection_t(self._U, self._V, self._T)[fluid_mask]
        )

        # Update the temperature field
        self._T = _Ttmp


    def calculate_temperatures_numba(self, grid):
        _Ttmp = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
        _Ttmp = _calculate_temperatures_numba(self._U, self._V, self._T, _Ttmp, self._dt, self._alpha, grid.get_fluid_cells_mask(), grid.dx(), grid.dy(), Discretization._gamma)
        self._T = _Ttmp


    def calculate_dt_naive(self, grid):
        # Implementation of adaptive timestep calculation
        umax = 0.0
        vmax = 0.0

        for elem in grid.fluid_cells():
            i = elem.i()
            j = elem.j()

            if abs(self._U[i, j]) > umax:
                umax = abs(self._U[i, j])
            if abs(self._V[i, j]) > vmax:
                vmax = abs(self._V[i, j])

            # print(f"U: {self._U[i, j]}, V: {self._V[i, j]}")

        arg1 = (1 / (2 * self._nu)) * (1 / ((1 / (grid.dx() ** 2)) + (1 / (grid.dy() ** 2))))
        arg2 = grid.dx() / umax
        arg3 = grid.dy() / vmax

        #print(f"arg1: {arg1}, arg2: {arg2}, arg3: {arg3}")

        self._dt = min(arg1, min(arg2, arg3))
        if grid.getUseTemp():
            arg4 = (1 / (2 * self._alpha)) * (1 / ((1 / (grid.dx() ** 2)) + (1 / (grid.dy() ** 2))))
            self._dt = min(self._dt, arg4)

        self._dt *= self._tau

        if self._dt < 0.0005:
            print(f"dt may get really small : {self._dt}")
            print(f"vmax: {vmax}, umax: {umax}")

        return self._dt
    
    
    def calculate_dt_vectorized(self, grid):
            # Adaptive timestep calculation

            fluid_mask = grid.get_fluid_cells_mask()

            # Compute umax and vmax using vectorized operations
            umax = np.max(np.abs(self._U))
            vmax = np.max(np.abs(self._V))

            # Compute arguments
            arg1 = (1 / (2 * self._nu)) * (1 / ((1 / (grid.dx() ** 2)) + (1 / (grid.dy() ** 2))))
            arg2 = grid.dx() / umax
            arg3 = grid.dy() / vmax  


            #print(f"arg1: {arg1}, arg2: {arg2}, arg3: {arg3}")


            # Calculate minimum timestep
            self._dt = min(arg1, min(arg2, arg3))

            # Adjust for temperature if applicable
            if grid.getUseTemp():
                arg4 = (1 / (2 * self._alpha)) * (1 / ((1 / (grid.dx() ** 2)) + (1 / (grid.dy() ** 2))))
                self._dt = min(self._dt, arg4)

            self._dt *= self._tau

            # Print a warning if dt is too small (optional)
            if self._dt < 0.0005:
                print(f"dt may get really small : {self._dt}")
                print(f"vmax: {vmax}, umax: {umax}")

            return self._dt
    

    
    """ 
    def calculate_dt_numba(self, grid: Grid):
            # Adaptive timestep calculation

            dx = grid.dx()
            dy = grid.dy()
            tau = self._tau
            dt = self._dt
            useTemp = grid.getUseTemp()
            U = self._U
            V = self._V
            nu = self._nu
            alpha = self._alpha

            self._dt = _calculate_dt_numba(dx, dy, tau, dt, U, V, nu, alpha, useTemp)

            return self._dt
     """

    def u(self, i, j):
        return self._U[i, j]
    
    def set_u(self, i, j, value):
        self._U[i, j] = value

    def v(self, i, j):
        return self._V[i, j]
    
    def set_v(self, i, j, value):
        self._V[i, j] = value

    def p(self, i, j):
        return self._P[i, j]
    
    def set_p(self, i, j, value):
        self._P[i, j] = value

    def t(self, i, j):
        return self._T[i, j]
    
    def set_t(self, i, j, value):
        self._T[i, j] = value

    def rs(self, i, j):
        return self._RS[i, j]
    
    def set_rs(self, i, j, value):
        self._RS[i, j] = value

    def f(self, i, j):
        return self._F[i, j]
    
    def set_f(self, i, j, value):
        self._F[i, j] = value

    def g(self, i, j):
        return self._G[i, j]

    def set_g(self, i, j, value):
        self._G[i, j] = value

    def dt(self):
        return self._dt

    def u_matrix(self):
        return self._U

    def v_matrix(self):
        return self._V

    def t_matrix(self):
        return self._T
    
    def set_t_matrix(self, T):
        self._T = T

    def p_matrix(self):
        return self._P

    def f_matrix(self):
        return self._F

    def g_matrix(self):
        return self._G
    
    def rs_matrix(self):
        return self._RS

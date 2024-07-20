import numpy as np
from numba import jit

class Discretization:
    _dx = 0.0
    _dy = 0.0
    _gamma = 0.0

    def __init__(self, dx, dy, gamma):
        Discretization._dx = dx
        Discretization._dy = dy
        Discretization._gamma = gamma



    @staticmethod
    @jit(nopython=True)
    def convection_u_numba(U, V, i, j, dx, dy, gamma):
        du2dx = (0.25 / dx) * (((U[i, j] + U[i + 1, j]) * (U[i, j] + U[i + 1, j])) - (U[i - 1, j] + U[i, j]) * (U[i - 1, j] + U[i, j]))
        du2dx += (gamma * 0.25 / dx) * (np.abs(U[i, j] + U[i + 1, j]) * (U[i, j] - U[i + 1, j]) - np.abs(U[i - 1, j] + U[i, j]) * (U[i - 1, j] - U[i, j]))

        duvdy = (0.25 / dy) * ((V[i, j] + V[i + 1, j]) * (U[i, j] + U[i, j + 1]) - (V[i, j - 1] + V[i + 1, j - 1]) * (U[i, j - 1] + U[i, j]))
        duvdy += (gamma * 0.25 / dy) * (np.abs(V[i, j] + V[i + 1, j]) * (U[i, j] - U[i, j + 1]) - np.abs(V[i, j - 1] + V[i + 1, j - 1]) * (U[i, j - 1] - U[i, j]))

        return du2dx + duvdy

    @staticmethod
    @jit(nopython=True)
    def convection_v_numba(U, V, i, j, dx, dy, gamma):
        dv2dy = (0.25 / dy) * ((V[i, j] + V[i, j + 1]) * (V[i, j] + V[i, j + 1]) - (V[i, j - 1] + V[i, j]) * (V[i, j - 1] + V[i, j]))
        dv2dy += (gamma * 0.25 / dy) * (np.abs(V[i, j] + V[i, j + 1]) * (V[i, j] - V[i, j + 1]) - np.abs(V[i, j - 1] + V[i, j]) * (V[i, j - 1] - V[i, j]))

        duvdx = (0.25 / dx) * ((V[i, j] + V[i + 1, j]) * (U[i, j] + U[i, j + 1]) - (V[i - 1, j] + V[i, j]) * (U[i - 1, j] + U[i - 1, j + 1]))
        duvdx += (gamma * 0.25 / dx) * (np.abs(U[i, j] + U[i, j + 1]) * (V[i, j] - V[i + 1, j]) - np.abs(U[i - 1, j] + U[i - 1, j + 1]) * (V[i - 1, j] - V[i, j]))

        return duvdx + dv2dy

    @staticmethod
    @jit(nopython=True)
    def convection_t_numba(U, V, T, i, j, dx, dy, gamma):
        duTdx = U[i, j] * (T[i, j] + T[i + 1, j]) - U[i - 1, j] * (T[i - 1, j] + T[i, j])
        duTdx += gamma * (np.abs(U[i, j]) * (T[i, j] - T[i + 1, j]) - np.abs(U[i - 1, j]) * (T[i - 1, j] - T[i, j]))
        duTdx *= (0.5 / dx)

        dvTdy = V[i, j] * (T[i, j] + T[i, j + 1]) - V[i, j - 1] * (T[i, j - 1] + T[i, j])
        dvTdy += gamma * (np.abs(V[i, j]) * (T[i, j] - T[i, j + 1]) - np.abs(V[i, j - 1]) * (T[i, j - 1] - T[i, j]))
        dvTdy *= (0.5 / dy)

        return duTdx + dvTdy


    @staticmethod
    @jit(nopython=True)
    def sor_helper_numba(P, i, j, dx, dy):
        return (P[i + 1, j] + P[i - 1, j]) / (dx ** 2) + (P[i, j + 1] + P[i, j - 1]) / (dy ** 2)


    @staticmethod
    def convection_u(U, V, i, j):
        du2dx = (0.25 / Discretization._dx) * (((U[i, j] + U[i + 1, j]) * (U[i, j] + U[i + 1, j])) - (U[i - 1, j] + U[i, j]) * (
                    U[i - 1, j] + U[i, j]))
        du2dx += (Discretization._gamma * 0.25 / Discretization._dx) * (
                    np.abs(U[i, j] + U[i + 1, j]) * (U[i, j] - U[i + 1, j]) - np.abs(U[i - 1, j] + U[i, j]) * (
                        U[i - 1, j] - U[i, j]))

        duvdy = (0.25 / Discretization._dy) * (
                    (V[i, j] + V[i + 1, j]) * (U[i, j] + U[i, j + 1]) - (V[i, j - 1] + V[i + 1, j - 1]) * (
                        U[i, j - 1] + U[i, j]))
        duvdy += (Discretization._gamma * 0.25 / Discretization._dy) * (
                    np.abs(V[i, j] + V[i + 1, j]) * (U[i, j] - U[i, j + 1]) - np.abs(V[i, j - 1] + V[i + 1, j - 1]) * (
                        U[i, j - 1] - U[i, j]))

        linear_combination = du2dx + duvdy

        return linear_combination
    

    @staticmethod
    def optimized_convection_u(U, V, grid):

        fluid_mask = grid.get_fluid_cells_mask()


        # Shift U horizontally
        U_right = np.roll(U, -1, axis=0)
        U_left = np.roll(U, 1, axis=0)
        
        # Calculate du2dx
        du2dx = (0.25 / Discretization._dx) * ((U + U_right) ** 2 - (U_left + U) ** 2)
        du2dx += (Discretization._gamma * 0.25 / Discretization._dx) * (np.abs(U + U_right) * (U - U_right) - np.abs(U_left + U) * (U_left - U))
        
        # Shift U and V vertically
        U_up = np.roll(U, -1, axis=1)
        V_up = np.roll(V, -1, axis=1)
        V_down = np.roll(V, 1, axis=1)
        V_right = np.roll(V, -1, axis=0)
        V_right_down = np.roll(V_right, 1, axis=1)
        U_down = np.roll(U, 1, axis=1)
        
        # Calculate duvdy
        duvdy = (0.25 / Discretization._dy) * ((V + V_right) * (U + U_up) - (V_down + V_right_down) * (U_down + U))
        duvdy += (Discretization._gamma * 0.25 / Discretization._dy) * (np.abs(V + V_right) * (U - U_up) - np.abs(V_down + V_right_down) * (U_down - U))
        
        # Combine du2dx and duvdy
        linear_combination = du2dx + duvdy

        return linear_combination
    



    @staticmethod
    def convection_v(U, V, i, j):
        dv2dy = (V[i, j] + V[i, j + 1]) * (V[i, j] + V[i, j + 1]) - (V[i, j - 1] + V[i, j]) * (V[i, j - 1] + V[i, j])
        dv2dy += Discretization._gamma * (np.abs(V[i, j] + V[i, j + 1]) * (V[i, j] - V[i, j + 1]) - np.abs(V[i, j - 1] + V[i, j]) * (
                    V[i, j - 1] - V[i, j]))
        dv2dy *= (0.25 / Discretization._dy)

        duvdx = (V[i, j] + V[i + 1, j]) * (U[i, j] + U[i, j + 1]) - (V[i - 1, j] + V[i, j]) * (
                    U[i - 1, j] + U[i - 1, j + 1])
        duvdx += Discretization._gamma * (
                    np.abs(U[i, j] + U[i, j + 1]) * (V[i, j] - V[i + 1, j]) - np.abs(U[i - 1, j] + U[i - 1, j + 1]) * (
                        V[i - 1, j] - V[i, j]))
        duvdx *= (0.25 / Discretization._dx)

        linear_combination = duvdx + dv2dy

        return linear_combination

    @staticmethod
    def optimized_convection_v(U, V, grid):
        fluid_mask = grid.get_fluid_cells_mask()

        # U_boundary = U[~fluid_mask]
        # V_boundary = V[~fluid_mask]

        # Shift V vertically
        V_up = np.roll(V, -1, axis=1)
        V_down = np.roll(V, 1, axis=1)

        # Calculate dv2dy
        dv2dy = (V + V_up) ** 2 - (V_down + V) ** 2
        dv2dy += Discretization._gamma * (np.abs(V + V_up) * (V - V_up) - np.abs(V_down + V) * (V_down - V))
        dv2dy *= 0.25 / Discretization._dy

        # Shift U horizontally and V vertically for duvdx calculation
        U_right = np.roll(U, -1, axis=0)
        U_left = np.roll(U, 1, axis=0)
        U_up = np.roll(U, -1, axis=1)
        V_right = np.roll(V, -1, axis=0)
        V_left = np.roll(V, 1, axis=0)

        U_left_up = np.roll(U_left, -1, axis=1)

        # Calculate duvdx
        duvdx = (V + V_right) * (U + U_up) - (V_left + V) * (U_left + U_left_up)
        duvdx += Discretization._gamma * (np.abs(U + U_up) * (V - V_right) - np.abs(U_left + U_left_up) * (V_left - V))
        duvdx *= 0.25 / Discretization._dx

        # Combine duvdx and dv2dy
        linear_combination = duvdx + dv2dy

        # Handling the boundaries
        # U[~fluid_mask] = U_boundary
        # V[~fluid_mask] = V_boundary

        return linear_combination







    @staticmethod
    def convection_u2(U, V):
        du2dx = (0.25 / Discretization._dx) * (
            (U[:-1, :] + U[1:, :])**2 - (U[:-2, :] + U[1:-1, :])**2 +
            Discretization._gamma * (np.abs(U[:-1, :] + U[1:, :]) * (U[:-1, :] - U[1:, :]) - 
                                    np.abs(U[:-2, :] + U[1:-1, :]) * (U[:-2, :] - U[1:-1, :]))
        )
        
        duvdy = (0.25 / Discretization._dy) * (
            (V[:, :-1] + V[:, 1:]) * (U[:, :-1] + U[:, 1:]) - 
            (V[:, :-2] + V[:, 1:-1]) * (U[:, :-2] + U[:, :-1]) +
            Discretization._gamma * (np.abs(V[:, :-1] + V[:, 1:]) * (U[:, :-1] - U[:, 1:]) - 
                                    np.abs(V[:, :-2] + V[:, 1:-1]) * (U[:, :-2] - U[:, :-1]))
        )
        
        return du2dx + duvdy
    

    @staticmethod
    def convection_v2(U, V):
        dv2dy = (0.25 / Discretization._dy) * (
            (V[:, :-1] + V[:, 1:])**2 - (V[:, :-2] + V[1:-1, :])**2 +
            Discretization._gamma * (np.abs(V[:, :-1] + V[:, 1:]) * (V[:, :-1] - V[:, 1:]) - 
                                    np.abs(V[:, :-2] + V[1:-1, :]) * (V[:, :-2] - V[1:-1, :]))
        )
        
        duvdx = (0.25 / Discretization._dx) * (
            (V[:-1, :] + V[1:, :]) * (U[:, :-1] + U[:, 1:]) - 
            (V[:-2, :] + V[1:-1, :]) * (U[:, :-2] + U[:, 1:-1]) +
            Discretization._gamma * (np.abs(U[:, :-1] + U[:, 1:]) * (V[:-1, :] - V[1:, :]) - 
                                    np.abs(U[:, :-2] + U[1:-1, :]) * (V[:-2, :] - V[1:-1, :]))
        )
        
        return duvdx + dv2dy


































    @staticmethod
    def convection_t(U, V, T, i, j):
        duTdx = U[i, j] * (T[i, j] + T[i + 1, j]) - U[i - 1, j] * (T[i - 1, j] + T[i, j])
        duTdx += Discretization._gamma * (np.abs(U[i, j]) * (T[i, j] - T[i + 1, j]) - np.abs(U[i - 1, j]) * (T[i - 1, j] - T[i, j]))
        duTdx *= (0.5 / Discretization._dx)

        dvTdy = V[i, j] * (T[i, j] + T[i, j + 1]) - V[i, j - 1] * (T[i, j - 1] + T[i, j])
        dvTdy += Discretization._gamma * (np.abs(V[i, j]) * (T[i, j] - T[i, j + 1]) - np.abs(V[i, j - 1]) * (T[i, j - 1] - T[i, j]))
        dvTdy *= (0.5 / Discretization._dy)

        linear_combination = duTdx + dvTdy
        return linear_combination
    

    @staticmethod
    def optimized_convection_t(U, V, T):
        duTdx = U * (T + np.roll(T, -1, axis=0)) - np.roll(U, 1, axis=0) * (np.roll(T, 1, axis=0) + T)
        duTdx += Discretization._gamma * (np.abs(U) * (T - np.roll(T, -1, axis=0)) - np.abs(np.roll(U, 1, axis=0)) * (np.roll(T, 1, axis=0) - T))
        duTdx *= 0.5 / Discretization._dx

        dvTdy = V * (T + np.roll(T, -1, axis=1)) - np.roll(V, 1, axis=1) * (np.roll(T, 1, axis=1) + T)
        dvTdy += Discretization._gamma * (np.abs(V) * (T - np.roll(T, -1, axis=1)) - np.abs(np.roll(V, 1, axis=1)) * (np.roll(T, 1, axis=1) - T))
        dvTdy *= 0.5 / Discretization._dy

        linear_combination = duTdx + dvTdy
        return linear_combination
    

    @staticmethod
    def laplacian(A, i, j):
        return (A[i + 1, j] - 2 * A[i, j] + A[i - 1, j]) / (Discretization._dx ** 2) + \
               (A[i, j + 1] - 2 * A[i, j] + A[i, j - 1]) / (Discretization._dy ** 2)


    @staticmethod
    def optimized_laplacian(A):  # We don't need i and j arguments here
        # Shifted arrays for efficient neighbor access (assuming valid padding)
        A_shifted_x_pos = np.roll(A, 1, axis=0)  # A[i - 1, j]
        A_shifted_x_neg = np.roll(A, -1, axis=0)  # A[i + 1, j]
        A_shifted_y_pos = np.roll(A, 1, axis=1)  # A[i, j - 1]
        A_shifted_y_neg = np.roll(A, -1, axis=1)  # A[i, j + 1]

        # Calculate laplacian terms using vectorized operations
        laplacian_terms = (A_shifted_x_neg - 2 * A + A_shifted_x_pos) / (Discretization._dx ** 2) + \
                          (A_shifted_y_neg - 2 * A + A_shifted_y_pos) / (Discretization._dy ** 2)


        return laplacian_terms
    
    @staticmethod
    @jit(nopython=True)
    def laplacian_numba(U, i, j, dx, dy):
        return (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1] - 4 * U[i, j]) / (dx * dy)


    @staticmethod
    def sor_helper(P, i, j):
        return (P[i + 1, j] + P[i - 1, j]) / (Discretization._dx ** 2) + \
               (P[i, j + 1] + P[i, j - 1]) / (Discretization._dy ** 2)
    

    @staticmethod
    def sor_helper2(P, i, j):
        # Initialize arrays for shifted indices
        i_plus_1 = i + 1
        i_minus_1 = i - 1
        j_plus_1 = j + 1
        j_minus_1 = j - 1

        # Calculate the SOR helper term using vectorized operations
        term_x = (P[i_plus_1, j] + P[i_minus_1, j]) / (Discretization._dx ** 2)
        term_y = (P[i, j_plus_1] + P[i, j_minus_1]) / (Discretization._dy ** 2)

        return term_x + term_y
    
    @staticmethod
    def optimized_sor_helper(P):
        # Shift P in all directions
        P_shifted_right = np.roll(P, -1, axis=0)
        P_shifted_left = np.roll(P, 1, axis=0)
        P_shifted_up = np.roll(P, -1, axis=1)
        P_shifted_down = np.roll(P, 1, axis=1)

        # Calculate the helper function
        sor_helper_matrix = (P_shifted_right + P_shifted_left) / (Discretization._dx ** 2) + \
                            (P_shifted_up + P_shifted_down) / (Discretization._dy ** 2)

        return sor_helper_matrix

    @staticmethod
    def interpolate(A, i, j, i_offset, j_offset):
        return 0.5 * (A[i, j] + A[i + i_offset, j + j_offset])



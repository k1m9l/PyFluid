from abc import ABC, abstractmethod
from typing import List, Dict
from Fields import Fields
from Cell import Cell
from Enums import BorderPosition
import numpy as np
from numba import jit

class Boundary(ABC):



    def __init__(self, cells: List[Cell]):
        self._cells = cells
        self._cell_info = np.array([
            [cell.i(), cell.j()] + cell.border_info()
            for cell in cells
        ])
        self._i = self._cell_info[:, 0]
        self._j = self._cell_info[:, 1]
        self._top = self._cell_info[:, 2]
        self._bottom = self._cell_info[:, 3]
        self._left = self._cell_info[:, 4]
        self._right = self._cell_info[:, 5]


    @abstractmethod
    def applyVelocity(self, field: Fields):
        pass


    @abstractmethod
    def applyPressure(self, field: Fields):
        pass


    def optimized_applyFlux(self, field: Fields):
        i = self._i
        j = self._j
        top = self._top
        bottom = self._bottom
        left = self._left
        right = self._right

        u = field._U
        v = field._V

        # Top border
        top_right = np.logical_and(top, right)
        top_left = np.logical_and(top, left)
        for idx in np.where(top_right)[0]:
            field._F[i[idx], j[idx]] = u[i[idx], j[idx]]
        for idx in np.where(top_left)[0]:
            field._F[i[idx] - 1, j[idx]] = u[i[idx] - 1, j[idx]]
        for idx in np.where(top)[0]:
            field._G[i[idx], j[idx]] = v[i[idx], j[idx]]

        # Bottom border
        bottom_right = np.logical_and(bottom, right)
        bottom_left = np.logical_and(bottom, left)
        for idx in np.where(bottom_right)[0]:
            field._F[i[idx], j[idx]] = u[i[idx], j[idx]]
        for idx in np.where(bottom_left)[0]:
            field._F[i[idx] - 1, j[idx]] = u[i[idx] - 1, j[idx]]
        for idx in np.where(bottom)[0]:
            field._G[i[idx], j[idx] - 1] = v[i[idx], j[idx] - 1]

        # Right border
        for idx in np.where(right)[0]:
            field._F[i[idx], j[idx]] = u[i[idx], j[idx]]

        # Left border
        for idx in np.where(left)[0]:
            field._F[i[idx] - 1, j[idx]] = u[i[idx] - 1, j[idx]]


    def applyFlux(self, field: Fields):

        u = field._U
        v = field._V

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            if elem.is_border(BorderPosition.TOP):
                if elem.is_border(BorderPosition.RIGHT):
                    field.set_f(i,j, u[i,j])

                elif elem.is_border(BorderPosition.LEFT):
                    field.set_f(i-1,j, u[i-1,j])

                field.set_g(i,j, v[i,j])

            elif elem.is_border(BorderPosition.BOTTOM):
                if elem.is_border(BorderPosition.RIGHT):
                    field.set_f(i,j, u[i,j])

                elif elem.is_border(BorderPosition.LEFT):
                    field.set_f(i-1,j, u[i-1,j])

                field.set_g(i,j-1, v[i,j-1])

            elif elem.is_border(BorderPosition.RIGHT):
                field.set_f(i,j, u[i,j])

            elif elem.is_border(BorderPosition.LEFT):
                field.set_f(i-1,j, u[i-1,j])

        
class FixedWallBoundary(Boundary):

    def __init__(self, cells: List[Cell], wall_temperature: Dict[int, float] = None):
        super().__init__(cells)
        self._wall_temperature = wall_temperature if wall_temperature is not None else {}

    def applyVelocity_vectorized(self, field: Fields):
        i = self._i
        j = self._j
        top = self._top
        bottom = self._bottom
        left = self._left
        right = self._right

        u = field._U
        v = field._V

        # Top border
        top_right = np.logical_and(top, right)
        top_left = np.logical_and(top, left)
        top_bottom = np.logical_and(top, bottom)

        for idx in np.where(top_right)[0]:
            field._U[i[idx], j[idx]] = 0
            field._V[i[idx], j[idx]] = 0
            field._U[i[idx] - 1, j[idx]] = -u[i[idx] - 1, j[idx] + 1]
            field._V[i[idx], j[idx] - 1] = -v[i[idx] + 1, j[idx] - 1]

        for idx in np.where(top_left)[0]:
            field._U[i[idx] - 1, j[idx]] = 0
            field._V[i[idx], j[idx]] = 0
            field._U[i[idx], j[idx]] = -u[i[idx], j[idx] + 1]
            field._V[i[idx], j[idx] - 1] = -v[i[idx] - 1, j[idx] - 1]

        for idx in np.where(top_bottom)[0]:
            field._U[i[idx], j[idx]] = -u[i[idx], j[idx] - 1]
            field._V[i[idx], j[idx]] = 0
            field._V[i[idx], j[idx] - 1] = 0
            field._U[i[idx] - 1, j[idx]] = -(u[i[idx] - 1, j[idx] - 1] + u[i[idx] - 1, j[idx] + 1]) / 2

        for idx in np.where(top & ~top_right & ~top_left & ~top_bottom)[0]:
            field._U[i[idx], j[idx]] = -u[i[idx], j[idx] + 1]
            field._V[i[idx], j[idx]] = 0

        # Bottom border
        bottom_right = np.logical_and(bottom, right)
        bottom_left = np.logical_and(bottom, left)

        for idx in np.where(bottom_right)[0]:
            field._U[i[idx], j[idx]] = 0
            field._V[i[idx], j[idx] - 1] = 0
            field._U[i[idx] - 1, j[idx]] = -u[i[idx] - 1, j[idx] - 1]
            field._V[i[idx], j[idx]] = -v[i[idx] + 1, j[idx]]

        for idx in np.where(bottom_left)[0]:
            field._U[i[idx] - 1, j[idx]] = 0
            field._V[i[idx], j[idx] - 1] = 0
            field._U[i[idx], j[idx]] = -u[i[idx], j[idx] - 1]
            field._V[i[idx], j[idx]] = -v[i[idx] - 1, j[idx]]

        for idx in np.where(bottom & ~bottom_right & ~bottom_left)[0]:
            field._U[i[idx], j[idx]] = -u[i[idx], j[idx] - 1]
            field._V[i[idx], j[idx] - 1] = 0

        # Right border
        right_left = np.logical_and(right, left)

        for idx in np.where(right_left)[0]:
            field._U[i[idx], j[idx]] = 0
            field._U[i[idx] - 1, j[idx]] = 0
            field._V[i[idx], j[idx]] = -(v[i[idx] + 1, j[idx]] + v[i[idx] - 1, j[idx]]) / 2

        for idx in np.where(right & ~right_left)[0]:
            field._U[i[idx], j[idx]] = 0
            field._V[i[idx], j[idx]] = -v[i[idx] + 1, j[idx]]

        # Left border
        for idx in np.where(left & ~right_left)[0]:
            field._U[i[idx] - 1, j[idx]] = 0
            field._V[i[idx], j[idx]] = -v[i[idx] - 1, j[idx]]

    def applyVelocity(self, field: Fields):

        u = field._U
        v = field._V

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            if (elem.is_border(BorderPosition.TOP)):

                # NE corner
                if (elem.is_border(BorderPosition.RIGHT)):
                    field.set_u(i, j, 0)
                    field.set_v(i, j, 0)
                    field.set_u(i-1, j, -u[i-1, j+1])
                    field.set_v(i, j-1, -v[i+1, j-1])
                
                # NW corner
                elif (elem.is_border(BorderPosition.LEFT)):

                    field.set_u(i-1, j, 0)
                    field.set_v(i, j, 0)
                    field.set_u(i, j, -u[i, j+1])
                    field.set_v(i, j-1, -v[
                        i-1, j-1])

                # Cells only having TOP boundary
                else:

                    field.set_u(i, j, -u[i, j+1])
                    field.set_v(i, j, 0)
            
            elif (elem.is_border(BorderPosition.BOTTOM)):

                # SE corner
                if (elem.is_border(BorderPosition.RIGHT)):
                    field.set_u(i, j, 0)
                    field.set_v(i, j-1, 0)
                    field.set_u(i-1, j, - u[i-1, j-1])
                    field.set_v(i, j, -v[i+1, j])
                
                # SW corner
                elif (elem.is_border(BorderPosition.LEFT)):
                    field.set_u(i-1, j, 0)
                    field.set_v(i, j-1, 0)
                    field.set_u(i, j, -u[i, j-1])
                    field.set_v(i, j, -v[i-1, j])
                
                # Cells only having BOTTOM boundary
                else:
                    field.set_u(i, j, -u[i, j-1])
                    field.set_v(i, j-1, 0)
                
            elif (elem.is_border(BorderPosition.RIGHT)):
                
                field.set_u(i, j, 0)
                field.set_v(i, j, -v[i+1, j])
                
            # Cells only having LEFT boundary
            elif (elem.is_border(BorderPosition.LEFT)):
                field.set_u(i-1, j, 0)
                field.set_v(i, j, -v[i-1, j])
        

    def applyPressure_vectorized(self, field: Fields):
        i = self._i
        j = self._j
        top = self._top
        bottom = self._bottom
        left = self._left
        right = self._right

        p = field._P

        # Top border
        top_right = np.logical_and(top, right)
        top_left = np.logical_and(top, left)
        top_bottom = np.logical_and(top, bottom)

        for idx in np.where(top_right)[0]:
            field._P[i[idx], j[idx]] = (p[i[idx], j[idx] + 1] + p[i[idx] + 1, j[idx]]) / 2

        for idx in np.where(top_left)[0]:
            field._P[i[idx], j[idx]] = (p[i[idx], j[idx] + 1] + p[i[idx] - 1, j[idx]]) / 2

        for idx in np.where(top_bottom)[0]:
            field._P[i[idx], j[idx]] = (p[i[idx], j[idx] + 1] + p[i[idx], j[idx] - 1]) / 2

        for idx in np.where(top & ~top_right & ~top_left & ~top_bottom)[0]:
            field._P[i[idx], j[idx]] = p[i[idx], j[idx] + 1]

        # Bottom border
        bottom_right = np.logical_and(bottom, right)
        bottom_left = np.logical_and(bottom, left)

        for idx in np.where(bottom_right)[0]:
            field._P[i[idx], j[idx]] = (p[i[idx] + 1, j[idx]] + p[i[idx], j[idx] - 1]) / 2

        for idx in np.where(bottom_left)[0]:
            field._P[i[idx], j[idx]] = (p[i[idx], j[idx] - 1] + p[i[idx] - 1, j[idx]]) / 2

        for idx in np.where(bottom & ~bottom_right & ~bottom_left)[0]:
            field._P[i[idx], j[idx]] = p[i[idx], j[idx] - 1]

        # Right border
        right_left = np.logical_and(right, left)

        for idx in np.where(right_left)[0]:
            field._P[i[idx], j[idx]] = (p[i[idx] + 1, j[idx]] + p[i[idx] - 1, j[idx]]) / 2

        for idx in np.where(right & ~right_left)[0]:
            field._P[i[idx], j[idx]] = p[i[idx] + 1, j[idx]]

        # Left border
        for idx in np.where(left & ~right_left)[0]:
            field._P[i[idx], j[idx]] = p[i[idx] - 1, j[idx]]
    
    def applyPressure(self, field: Fields):

        p = field._P

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            if (elem.is_border(BorderPosition.TOP)):

                if (elem.is_border(BorderPosition.RIGHT)):
                    field.set_p(i, j, (p[i, j+1] + p[i+1, j]) / 2)
                elif (elem.is_border(BorderPosition.LEFT)):
                    field.set_p(i, j, (p[i, j+1] + p[i-1, j]) / 2)
                elif (elem.is_border(BorderPosition.BOTTOM)):
                    field.set_p(i, j, (p[i, j+1] + p[i, j-1]) / 2)
                else:
                    field.set_p(i, j, p[i, j+1])
                
            elif (elem.is_border(BorderPosition.BOTTOM)):

                if (elem.is_border(BorderPosition.RIGHT)):
                    field.set_p(i, j, (p[i+1, j] + p[i, j-1]) / 2)
                elif (elem.is_border(BorderPosition.LEFT)):
                    field.set_p(i, j, (p[i, j-1] + p[i-1, j]) / 2)
                else:
                    field.set_p(i, j, p[i, j-1])
                
            elif (elem.is_border(BorderPosition.RIGHT)):

                if (elem.is_border(BorderPosition.LEFT)):
                    field.set_p(i, j, (p[i+1, j] + p[i-1, j]) / 2)
                else:
                    field.set_p(i, j, p[i+1, j])
                
            elif (elem.is_border(BorderPosition.LEFT)):
                field.set_p(i, j, p[i-1, j])

    def applyTemperature_vectorized(self, field):
        wall_type = next(iter(self._wall_temperature.keys()))
        td = next(iter(self._wall_temperature.values()))

        i = self._i
        j = self._j
        top = self._top
        bottom = self._bottom
        left = self._left
        right = self._right

        t = field._T

        # Top border
        top_right = np.logical_and(top, right)
        top_left = np.logical_and(top, left)
        top_bottom = np.logical_and(top, bottom)

        for idx in np.where(top_right)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - (t[i[idx], j[idx] + 1] + t[i[idx] + 1, j[idx]]) * 0.5
            else:
                field._T[i[idx], j[idx]] = 0.5 * (t[i[idx], j[idx] + 1] + t[i[idx] + 1, j[idx]])

        for idx in np.where(top_left)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - (t[i[idx], j[idx] + 1] + t[i[idx] - 1, j[idx]]) * 0.5
            else:
                field._T[i[idx], j[idx]] = 0.5 * (t[i[idx], j[idx] + 1] + t[i[idx] - 1, j[idx]])

        for idx in np.where(top_bottom)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - (t[i[idx], j[idx] + 1] + t[i[idx], j[idx] - 1]) * 0.5
            else:
                field._T[i[idx], j[idx]] = 0.5 * (t[i[idx], j[idx] + 1] + t[i[idx], j[idx] - 1])

        for idx in np.where(top & ~top_right & ~top_left & ~top_bottom)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - t[i[idx], j[idx] + 1]
            else:
                field._T[i[idx], j[idx]] = t[i[idx], j[idx] + 1]

        # Bottom border
        bottom_right = np.logical_and(bottom, right)
        bottom_left = np.logical_and(bottom, left)

        for idx in np.where(bottom_right)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - (t[i[idx], j[idx] - 1] + t[i[idx] + 1, j[idx]]) * 0.5
            else:
                field._T[i[idx], j[idx]] = 0.5 * (t[i[idx], j[idx] - 1] + t[i[idx] + 1, j[idx]])

        for idx in np.where(bottom_left)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - (t[i[idx], j[idx] - 1] + t[i[idx] - 1, j[idx]]) * 0.5
            else:
                field._T[i[idx], j[idx]] = 0.5 * (t[i[idx], j[idx] - 1] + t[i[idx] - 1, j[idx]])

        for idx in np.where(bottom & ~bottom_right & ~bottom_left)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - t[i[idx], j[idx] - 1]
            else:
                field._T[i[idx], j[idx]] = t[i[idx], j[idx] - 1]

        # Left border
        left_right = np.logical_and(left, right)

        for idx in np.where(left_right)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - 0.5 * (t[i[idx] - 1, j[idx]] + t[i[idx] + 1, j[idx]])
            else:
                field._T[i[idx], j[idx]] = 0.5 * (t[i[idx] - 1, j[idx]] + t[i[idx] + 1, j[idx]])

        for idx in np.where(left & ~left_right)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - t[i[idx] - 1, j[idx]]
            else:
                field._T[i[idx], j[idx]] = t[i[idx] - 1, j[idx]]

        # Right border
        for idx in np.where(right & ~left_right)[0]:
            if wall_type != 3:
                field._T[i[idx], j[idx]] = 2 * td - t[i[idx] + 1, j[idx]]
            else:
                field._T[i[idx], j[idx]] = t[i[idx] + 1, j[idx]]

                
    def applyTemperature(self, field):

        t = field._T
      
        # wall type
        wall_type = next(iter(self._wall_temperature.keys()))
        # wall temperature
        td = next(iter(self._wall_temperature.values()))

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            # Applying the temperature BCs according to the wall id
            if (elem.is_border(BorderPosition.TOP)):
                # For NE corner cell
                if (elem.is_border(BorderPosition.RIGHT)):
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - (t[i, j+1] + t[i+1, j]) * 0.5)
                    else :
                        field.set_t(i, j, 0.5 * (t[i, j+1] + t[i+1, j]))

                # For NW corner cell (Fluid is present in the north and west of this corner cell)
                elif (elem.is_border(BorderPosition.LEFT)):
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - (t[i, j+1] + t[i-1, j]) * 0.5)
                    else:
                        field.set_t(i, j, 0.5 * (t[i, j+1] + t[i-1, j]))
                
                #  (Fluid is present in both north and south of this cell)
                elif (elem.is_border(BorderPosition.BOTTOM)):
                    if wall_type != 3:
                        field.set_t(i, j, 2 * td - (t[i, j+1] + t[i, j-1]) * 0.5)
                    else:
                        field.set_t(i, j, 0.5 * (t[i, j+1] + t[i, j-1])) 
                
                # For bottommost boundary  (Fluid is present ONLY  in the north  of these cells)
                else :
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - t[i, j+1])
                    else :
                        field.set_t(i, j, t[i, j+1])
                    
            if (elem.is_border(BorderPosition.BOTTOM)) :

                # SE cell
                if (elem.is_border(BorderPosition.RIGHT)) :
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - (t[i, j-1] + t[i+1, j]) * 0.5)
                    else :
                        field.set_t(i, j, 0.5 * (t[i, j-1] + t[i+1, j]))
                    
                # For SW corner cell (Fluid is present in the north and west of this corner cell)
                elif (elem.is_border(BorderPosition.LEFT)) :
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - (t[i, j-1] + t[i-1, j]) * 0.5)
                    else :
                        field.set_t(i, j, 0.5 * (t[i, j-1] + t[i-1, j]))
                    
                # For topmost boundary
                else :
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - t[i, j-1])
                    else :
                        field.set_t(i, j, t[i, j-1])
                    
            if (elem.is_border(BorderPosition.LEFT)) :

                # Fluid cells exist on the left and right borders
                if (elem.is_border(BorderPosition.RIGHT)) :
                    if wall_type != 3 :
                        field.set_t(i, j,
                            2 * td - 0.5 * (t[i-1, j] + t[i+1, j]))
                    else :
                        field.set_t(i, j, 0.5 * (t[i-1, j] + t[i+1, j]))
                    
                # For rightmost boundary (Fluid cells exist on the left)
                else :
                    if wall_type != 3 :
                        field.set_t(i, j, 2 * td - t[i-1, j])
                    else :
                        field.set_t(i, j, t[i-1, j])
                    
            # For leftmost boundary
            if (elem.is_border(BorderPosition.RIGHT)) :
                if wall_type != 3 :
                    field.set_t(i, j, 2 * td - t[i+1, j])
                else :
                    field.set_t(i, j, t[i+1, j])

class MovingWallBoundary(Boundary):

    def __init__(self, cells: List[Cell], wall_velocity: Dict[int, float], wall_temperature: Dict[int, float] = None):
        super().__init__(cells)
        self._wall_velocity = wall_velocity
        self._wall_temperature = wall_temperature
    
    def applyVelocity(self, field):
        i = self._i
        j = self._j

        u = field._U
        v = field._V

        wall_velocity = self._wall_velocity

        field._U[i, j] = 2 * wall_velocity - u[i, j - 1]
        field._V[i, j - 1] = 0
    

    def applyPressure(self, field: Fields):
        i = self._i
        j = self._j

        p = field._P

        field._P[i, j] = p[i, j - 1]


class InflowBoundary(Boundary):

    def __init__(self, cells: List[Cell], inflow_u: float, inflow_v: float):
        super().__init__(cells)
        self._inflow_u = inflow_u
        self._inflow_v = inflow_v

    def applyVelocity_vectorized(self, field: Fields):
        i = self._i
        j = self._j

        v = field._V

        field._U[i, j] = self._inflow_u
        field._V[i, j] = - v[i + 1, j]

    def applyVelocity(self, field):
        u = field._U
        v = field._V

        for elem in self._cells:
            i = elem.i()
            j = elem.j()
            field.set_u(i, j, self._inflow_u);
            field.set_v(i, j, -v[i + 1, j])

    def applyPressure_vectorized(self, field: Fields):
        i = self._i
        j = self._j

        p = field._P

        field._P[i, j] = p[i + 1, j]

    def applyPressure(self, field: Fields):
        p = field._P

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            field.set_p(i, j, p[i + 1, j]);

           
class OutflowBoundary(Boundary):

    def __init__(self, cells: List[Cell], pressure: float):
        super().__init__(cells)
        self._pressure = pressure

    def applyVelocity_vectorized(self, field: Fields):
        i = self._i
        j = self._j
        left = self._left
        right = self._right

        u = field._U
        v = field._V

        left_indices = np.where(left)[0]
        right_indices = np.where(right)[0]

        field._U[i[left_indices], j[left_indices]] = u[i[left_indices] - 1, j[left_indices]]
        field._V[i[left_indices], j[left_indices]] = v[i[left_indices] - 1, j[left_indices]]

        field._U[i[right_indices], j[right_indices]] = u[i[right_indices] + 1, j[right_indices]]
        field._V[i[right_indices], j[right_indices]] = v[i[right_indices] + 1, j[right_indices]]

    def applyVelocity(self, field: Fields):

        u = field._U
        v = field._V

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            if (elem.is_border(BorderPosition.LEFT)):
                field.set_u(i, j, u[i-1, j])
                field.set_v(i, j, v[i-1, j])
            else :
                field.set_u(i, j, u[i+1, j])
                field.set_v(i, j, v[i+1, j])
            
    
    def applyPressure_vectorized(self, field: Fields):
        i = self._i
        j = self._j
        left = self._left
        right = self._right

        p = field._P

        left_indices = np.where(left)[0]
        right_indices = np.where(right)[0]

        field._P[i[left_indices], j[left_indices]] = 2 * self._pressure - p[i[left_indices] - 1, j[left_indices]]
        field._P[i[right_indices], j[right_indices]] = 2 * self._pressure - p[i[right_indices] + 1, j[right_indices]]

    def applyPressure(self, field: Fields):
        p = field._P

        for elem in self._cells:
            i = elem.i()
            j = elem.j()

            if elem.is_border(BorderPosition.LEFT) :
                field.set_p(i, j, 2 * self._pressure - p[i - 1, j])
        
            elif elem.is_border(BorderPosition.RIGHT) :
                field.set_p(i, j, 2 * self._pressure - p[i + 1, j])
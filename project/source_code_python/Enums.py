# If no geometry file is provided in the input file, lid driven cavity case
# will run by default. In the Grid.py, geometry will be created following
# PGM convention, which is:
# 0: fluid, 3: fixed wall, 4: moving wall

from enum import Enum

class LidDrivenCavity:
    moving_wall_id = 8
    fixed_wall_id = 4
    wall_velocity = 1.0


class BorderPosition(Enum):
    TOP = 1
    BOTTOM = 2
    LEFT = 3
    RIGHT = 4

class CellType(Enum):
    FLUID = 1
    INFLOW = 2
    OUTFLOW = 3
    FIXED_WALL = 4
    FIXED_WALL_HEAT = 5
    FIXED_WALL_COLD = 6
    MOVING_WALL = 7
    DEFAULT = 8



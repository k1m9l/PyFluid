from Enums import BorderPosition, CellType
from typing import List, Optional, Type
from collections import deque


class Cell:
    def __init__(self, i: int, j: int, cell_type: CellType, id=0):
            self._i = i
            self._j = j
            self._type = cell_type
            self._id = id
            # Initialize attributes that might be copied from another cell
            self._border = {position: False for position in BorderPosition}
            self._borders = deque()
            self._neighbours = {}


    def neighbour(self, position: BorderPosition):
        return self._neighbours[position]

    def set_neighbour(self, cell , position: BorderPosition):
        self._neighbours[position] = cell

    def borders(self) -> List[BorderPosition]:
        return self._borders

    def add_border(self, border: BorderPosition):
        self._border[border] = True
        self._borders.append(border)

    def is_border(self, position: BorderPosition) -> bool:
        return self._border[position]

    def i(self) -> int:
        return self._i

    def j(self) -> int:
        return self._j

    def type(self) -> CellType:
        return self._type

    def wall_id(self) -> int:
        return self._id

    def get_borders(self) -> List[bool]:
        return [False] * 4
    
    def border_info(self) -> List[bool]:
        return [self._border[position] for position in BorderPosition]

    


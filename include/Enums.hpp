#pragma once

// If no geometry file is provided in the input file, lid driven cavity case
// will run by default. In the Grid.cpp, geometry will be created following
// PGM convention, which is:
// 0: fluid, 3: fixed wall, 4: moving wall
namespace LidDrivenCavity {
const int moving_wall_id = 8;
const int fixed_wall_id = 4;
const double wall_velocity = 1.0;
} // namespace LidDrivenCavity

enum class border_position {
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,
};

enum class cell_type {
    FLUID,
    INFLOW,
    OUTFLOW,
    FIXED_WALL,
    FIXED_WALL_HEAT,
    FIXED_WALL_COLD,
    MOVING_WALL,
    DEFAULT,
};

#include "Boundary.hpp"
#include <iostream>
#include <algorithm>

Boundary::Boundary(std::vector<Cell *> cells) : _cells(cells) {}

void Boundary::applyFlux(Fields &field) {

    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        if (elem->is_border(border_position::TOP)) {
            if (elem->is_border(border_position::RIGHT)) {
                field.f(i,j) = field.u(i,j);
            }

            else if (elem->is_border(border_position::LEFT)) {
                field.f(i-1,j) = field.u(i-1,j);
            }

            field.g(i,j) = field.v(i,j);
        }

        else if (elem->is_border(border_position::BOTTOM)) {
            if (elem->is_border(border_position::RIGHT)) {
                field.f(i,j) = field.u(i,j);
            }

            else if (elem->is_border(border_position::LEFT)) {
                field.f(i-1,j) = field.u(i-1,j);
            }

            field.g(i,j-1) = field.v(i,j-1);
        }

        else if (elem->is_border(border_position::RIGHT)) {
            field.f(i,j) = field.u(i,j);
        }

        else if (elem->is_border(border_position::LEFT)) {
            field.f(i-1,j) = field.u(i-1,j);
        }
    }
}

void Boundary::validCell() {

    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        // check for opposing boundaries
        if ((elem->is_border(border_position::TOP) && elem->is_border(border_position::BOTTOM))
            || (elem->is_border(border_position::RIGHT) && elem->is_border(border_position::LEFT))) {
            throw std::invalid_argument("Opposing boundaries are not allowed at index: "
                                        + std::to_string(i) + ", " + std::to_string(j) + ". Adapt input file");
        }

        // check number of boundaries to be less than 3
        if (std::count(elem->getBorders().begin(), elem->getBorders().end(), true) > 2) {
           throw std::invalid_argument("Obstacle cannot have more than two boundaries at index: "
                                        + std::to_string(i) + ", " + std::to_string(j) + ". Adapt input file");
        }
    }
}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : Boundary(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : Boundary(cells), _wall_temperature(wall_temperature) {}

void FixedWallBoundary::applyVelocity(Fields &field) {

    for (auto &elem : _cells) {
        int i = elem -> i();
        int j = elem -> j();

        if (elem->is_border(border_position::TOP)) {
            if (elem->is_border(border_position::RIGHT)) {
                field.u(i, j) = 0;
                field.u(i-1, j) = -field.u(i-1, j+1);

                field.v(i, j) = 0;
                field.v(i, j-1) = -field.v(i+1, j-1);
            }

            else if (elem->is_border(border_position::LEFT)) {
                field.u(i,j) = -field.u(i,j+1);
                field.u(i-1,j) = 0;

                field.v(i,j) = 0;
                field.v(i,j-1) = -field.v(i-1, j-1);
            }

            else {
                field.u(i, j) = -field.u(i, j+1);
                field.v(i, j) = 0;
            }
        }

        else if (elem->is_border(border_position::BOTTOM)) {
            if (elem->is_border(border_position::RIGHT)) {
                field.u(i, j) = 0;
                field.u(i-1, j) = -field.u(i-1, j-1);

                field.v(i,j) = -field.v(i+1,j);
                field.v(i,j-1) = 0;
            }

            else if (elem->is_border(border_position::LEFT)) {
                field.u(i, j) = -field.u(i, j-1);
                field.u(i-1, j) = 0;

                field.v(i, j) = -field.v(i-1,j);
                field.v(i,j-1) = 0;
            }

            else {
                field.u(i, j) = -field.u(i, j-1);
                field.v(i, j) = 0;
            }
        }

        else if (elem->is_border(border_position::RIGHT)) {
            field.u(i,j) = 0;
            field.v(i,j) = -field.v(i+1,j);
        }

        else if (elem->is_border(border_position::LEFT)) {
            field.u(i-1, j) = 0;
            field.v(i,j) = -field.v(i-1,j);
        }
    }
}

void FixedWallBoundary::applyPressure(Fields &field) {

    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        if (elem->is_border(border_position::TOP)) {
            if (elem->is_border(border_position::RIGHT)) {
                field.p(i,j) = (field.p(i+1,j) + field.p(i,j+1)) / 2;
            }

            if (elem->is_border(border_position::LEFT)) {
                field.p(i,j) = (field.p(i-1,j) + field.p(i,j+1)) / 2;
            }

            else {
                field.p(i,j) = field.p(i,j+1);
            }
        }

        else if (elem->is_border(border_position::BOTTOM)) {
            if (elem->is_border(border_position::RIGHT)) {
                field.p(i,j) = (field.p(i+1,j) + field.p(i,j-1)) / 2;
            }

            else if (elem->is_border(border_position::LEFT)) {
                field.p(i,j) = (field.p(i-1,j) + field.p(i,j-1)) / 2;
            }

            else {
                field.p(i,j) = field.p(i,j-1);
            }
        }

        else if (elem->is_border(border_position::RIGHT)) {
            field.p(i,j) = field.p(i+1,j);
        }

        else if (elem->is_border(border_position::LEFT)) {
            field.p(i,j) = field.p(i-1,j);
        }
    }
}
void FixedWallBoundary::applyTemperature(Fields &field) {
    // we iterate and check the direction of the boundaries of the cells
    // furthermore, we need to check if the cell is heat, cold or adiabatic in each case

    // wall type
    int wall_type{_wall_temperature.begin()->first};
    // wall temperature
    double td{_wall_temperature.begin()->second};

    for (auto &elem : _cells) {
        int i{elem->i()};
        int j{elem->j()};

        if (elem->is_border(border_position::TOP)) {
            if (elem->is_border(border_position::RIGHT)) {
                if (wall_type != 3) {
                    // dirichlet boundary condition
                    field.t(i,j) = 2*td - (field.t(i+1,j) + field.t(i,j+1))/2;
                }
                else {
                    // adiabatic case/ von Neumann boundary condition
                    field.t(i,j) = (field.t(i+1,j) + field.t(i,j+1))/2;
                }
            }

            else if (elem->is_border(border_position::LEFT)) {
                if (wall_type != 3) {
                    // dirichlet boundary condition
                    field.t(i,j) = 2*td - (field.t(i-1,j) + field.t(i,j+1))/2;
                }
                else {
                    // adiabatic case/ von Neumann boundary condition
                    field.t(i,j) = (field.t(i-1,j) + field.t(i,j+1))/2;
                }
            }

            else {
                if (wall_type != 3) {
                    // dirichlet boundary condition
                    field.t(i,j) = 2*td - field.t(i,j+1);
                }
                else {
                    // adiabatic case/ von Neumann boundary condition
                    field.t(i,j) = field.t(i,j+1);
                }
            }
        }

        else if (elem->is_border(border_position::BOTTOM)) {
            if (elem->is_border(border_position::RIGHT)) {
                if (wall_type != 3) {
                    // dirichlet boundary condition
                    field.t(i,j) = 2*td - (field.t(i+1,j) + field.t(i,j-1))/2;
                }
                else {
                    // adiabatic case/ von Neumann boundary condition
                    field.t(i,j) = (field.t(i+1,j) + field.t(i,j-1))/2;
                }
            }

            else if (elem->is_border(border_position::LEFT)) {
                if (wall_type != 3) {
                    // dirichlet boundary condition
                    field.t(i,j) = 2*td - (field.t(i-1,j) + field.t(i,j-1))/2;
                }
                else {
                    // adiabatic case/ von Neumann boundary condition
                    field.t(i,j) = (field.t(i-1,j) + field.t(i,j-1))/2;
                }
            }

            else {
                if (wall_type != 3) {
                    // dirichlet boundary condition
                    field.t(i,j) = 2*td - field.t(i,j-1);
                }
                else {
                    // adiabatic case/ von Neumann boundary condition
                    field.t(i,j) = field.t(i,j-1);
                }
            }
        }

        else if (elem->is_border(border_position::RIGHT)) {
            if (wall_type != 3) {
                // dirichlet boundary condition
                field.t(i,j) = 2*td - field.t(i+1,j);
            }
            else {
                // adiabatic case/ von Neumann boundary condition
                field.t(i,j) = field.t(i+1,j);
            }
        }

        else if (elem->is_border(border_position::LEFT)) {
            if (wall_type != 3) {
                // dirichlet boundary condition
                field.t(i,j) = 2*td - field.t(i-1,j);
            }
            else {
                // adiabatic case/ von Neumann boundary condition
                field.t(i,j) = field.t(i-1,j);
            }
        }
    }
}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, double wall_velocity) : Boundary(cells) {
    _wall_velocity.insert(std::pair(LidDrivenCavity::moving_wall_id, wall_velocity));
}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_velocity,
                                       std::map<int, double> wall_temperature)
    : Boundary(cells), _wall_velocity(wall_velocity), _wall_temperature(wall_temperature) {}

void MovingWallBoundary::applyVelocity(Fields &field) {
    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        // bottom boundary
        if (elem -> is_border(border_position::TOP)) {
            field.u(i,j) = 2 * (_wall_velocity.begin()->second) - field.u(i,j+1);
            field.v(i,j) = 0;
        }

        // top boundary
        else if (elem -> is_border(border_position::BOTTOM)) {
            field.u(i,j) = 2 * (_wall_velocity.begin()->second) - field.u(i, j-1);
            field.v(i, j-1) = 0;
        }

        // left boundary
        else if (elem -> is_border(border_position::RIGHT)) {
            field.u(i, j) = 0;
            field.v(i,j) = 2 * (_wall_velocity.begin()->second) - field.v(i+1,j);
        }

        // right boundary
        else if (elem -> is_border(border_position::LEFT)) {
            field.u(i-1,j) = 0;
            field.v(i,j) = 2 * (_wall_velocity.begin()->second) - field.v(i-1,j);
        }
    }
}

void MovingWallBoundary::applyPressure(Fields &field) {

    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        // bottom boundary
        if (elem->is_border(border_position::TOP)) {
            field.p(i, j) = field.p(i,j+1);
        }

        // top boundary
        else if (elem->is_border(border_position::BOTTOM)) {
            field.p(i, j) = field.p(i,j-1);
        }

        // left boundary
        else if (elem->is_border(border_position::RIGHT)) {
            field.p(i, j) = field.p(i+1, j);
        }

        // right boundary
        if (elem->is_border(border_position::LEFT)) {
            field.p(i, j) = field.p(i-1, j);
        }
    }
}

InflowBoundary::InflowBoundary(std::vector<Cell *> cells, double inflow_u, double inflow_v)
    : Boundary(cells), _inflow_u(inflow_u), _inflow_v(inflow_v) {}

void InflowBoundary::applyVelocity(Fields &field) {
    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        // bottom boundary
        if (elem -> is_border(border_position::TOP)) {
            field.u(i,j) = 2 * _inflow_u - field.u(i,j+1);
            field.v(i,j) = _inflow_v;
        }

        // top boundary
        else if (elem -> is_border(border_position::BOTTOM)) {
            field.u(i,j) = 2 * _inflow_u - field.u(i, j-1);
            field.v(i, j-1) = _inflow_v;
        }

        // left boundary
        else if (elem -> is_border(border_position::RIGHT)) {
            field.u(i, j) = _inflow_u;
            field.v(i, j) = 2 * (_inflow_v)-field.v(i+1, j);
        }

        // right boundary
        else if (elem -> is_border(border_position::LEFT)) {
            field.u(i-1,j) = _inflow_u;
            field.v(i,j) = 2 * _inflow_v - field.v(i-1,j);
        }
    }
}
void InflowBoundary::applyPressure(Fields &field) {
    for (auto &elem : _cells) {
        int i = elem->i();
        int j = elem->j();

        // bottom boundary
        if (elem->is_border(border_position::TOP)) {
            field.p(i, j) = field.p(i,j+1);
        }

        // top boundary
        else if (elem->is_border(border_position::BOTTOM)) {
            field.p(i, j) = field.p(i,j-1);
        }

        // left boundary
        else if (elem->is_border(border_position::RIGHT)) {
            field.p(i, j) = field.p(i+1, j);
        }

        // right boundary
        if (elem->is_border(border_position::LEFT)) {
            field.p(i, j) = field.p(i-1, j);
        }
    }
}

OutflowBoundary::OutflowBoundary(std::vector<Cell *> cells, double pressure)
    : Boundary(cells), _pressure(pressure) {}

void OutflowBoundary::applyVelocity(Fields &field) {
    for (const auto& elem : _cells) {
        auto i{elem->i()};
        auto j{elem->j()};

        // bottom boundary
        if (elem->is_border(border_position::TOP)) {
            field.u(i,j) = field.u(i,j+1);
            field.v(i,j) = field.v(i,j+1);
        }

        // top boundary
        if (elem->is_border(border_position::BOTTOM)) {
            field.u(i,j) = field.u(i,j-1);
            field.v(i,j) = field.v(i,j-1);
        }

        // left boundary
        if (elem->is_border(border_position::RIGHT)) {
            field.u(i,j) = field.u(i+1,j);
            field.v(i,j) = field.v(i+1,j);
        }

        // right boundary
        if (elem->is_border(border_position::LEFT)) {
            field.u(i,j) = field.u(i-1,j);
            field.v(i,j) = field.v(i-1,j);
        }
    }
}

void OutflowBoundary::applyPressure(Fields &field) {
    for (const auto& elem : _cells) {
        auto i{elem->i()};
        auto j{elem->j()};

        // outflow bottom
        if (elem->is_border(border_position::TOP)) {
            field.p(i,j) = 2*_pressure - field.p(i, j+1);
        }

        // outflow top
        if (elem->is_border(border_position::BOTTOM)) {
            field.p(i,j) = 2*_pressure - field.p(i, j-1);
        }

        // outflow right
        if (elem->is_border(border_position::LEFT)) {
            field.p(i,j) = 2*_pressure - field.p(i-1, j);
        }

        // outflow left
        if (elem->is_border(border_position::RIGHT)) {
            field.p(i,j) = 2*_pressure - field.p(i+1, j);
        }
    }
}

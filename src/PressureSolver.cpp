#include <cmath>

#include "Communication.hpp"
#include "PressureSolver.hpp"

SOR::SOR(double omega) : _omega(omega) {}

double SOR::solve(Fields &field, Grid &grid) {

    double dx = grid.dx();
    double dy = grid.dy();

    double coeff = _omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy))); // = _omega * h^2 / 4.0, if dx == dy == h

    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        // Exclude cells for communication
        if (i != 0 && i != grid.size_x() + 1 && j != 0 && j != grid.size_y() + 1) {
            field.p(i, j) = (1.0 - _omega) * field.p(i, j) + coeff * (Discretization::sor_helper(field.p_matrix(), i, j) - field.rs(i, j));
        }
    }

    double rloc = 0.0;

    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();

        // Exclude cells for communication
        if (i != 0 && i != grid.size_x() + 1 && j != 0 && j != grid.size_y() + 1) {
            double val = Discretization::laplacian(field.p_matrix(), i, j) - field.rs(i, j);
            rloc += (val * val);
        }
    }
    //{
    //    res = rloc / (grid.fluid_cells().size());
    //    res = std::sqrt(res);
    //}

    return rloc;
}

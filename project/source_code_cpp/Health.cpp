#include "Health.h"

bool Health::isHealthy(Fields& field, Grid& grid) {
    return !(checkVelocity(field, grid) && checkPressure(field, grid));
}

bool Health::checkVelocity(Fields &field, Grid &grid) {

    for (auto& elem : grid.fluid_cells()) {

        auto tmpU = field.u(elem->i(), elem->j());
        if (std::isnan(tmpU) || std::isinf(tmpU)) {
            return true;
        }
        auto tmpV = field.v(elem->i(), elem->j());
        if (std::isnan(tmpV) || std::isinf(tmpV)) {
            return true;
        }
    }
    return false;
}

bool Health::checkPressure(Fields& field, Grid& grid) {
    for (auto& elem : grid.fluid_cells()) {
        if (std::isnan(field.p(elem->i(), elem->j())) || std::isinf(field.p(elem->i(), elem->j()))) {
            return true;
        }
    }
    return false;
}
#ifndef CFDLAB_HEALTH_H
#define CFDLAB_HEALTH_H

#include <limits>
#include "Fields.hpp"
#include "Grid.hpp"
#include <iostream>
#include <math.h>

/**
 * @brief Class to hold and orchestrate the simulation flow.
 *
 */
class Health {
public:
    /**
     * @brief Constructor for the Health class
     *
     */
    Health()=default;

    /**
     * @brief public method to check if the simulation is still in a healthy state
     *
     * @param[in] Fields field
     * @param[in] Grid grid
     * @param[out] boolean
     *
     */
    bool isHealthy(Fields& field, Grid& grid);

private:

    /**
     * @brief helper method to determine if a value in the velocities matrix is nan or inf
     *
     * @param[in] Fields field
     * @param[in] Grid grid
     * @param[out] bool
     *
     */
    bool checkVelocity(Fields& field, Grid& grid);

    /**
     * @brief helper method to determine if a value in the pressure matrix is nan or inf
     *
     * @param[in] Fields field
     * @param[in] Grid grid
     * @param[out] bool
     *
     */
    bool checkPressure(Fields& field, Grid& grid);
};
#endif //CFDLAB_HEALTH_H

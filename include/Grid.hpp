#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Cell.hpp"
#include "Datastructures.hpp"
#include "Domain.hpp"
#include "Enums.hpp"

/**
 * @brief Data structure holds cells and related sub-containers
 *
 */
class Grid {
  public:
    Grid() = default;

    /**
     * @brief Constructor for the Grid
     *
     * @param[in] geom_name geometry file name
     * @param[in] domain struct storing geometry information
     *
     */
    Grid(std::string geom_name, Domain &domain, int iProc, int jProc, int rank, int size);

    /// index based cell access
    Cell cell(int i, int j) const;

    /// access number of cells in x direction
    int size_x() const;
    /// access number of cells in y direction
    int size_y() const;

    /// access number of cells in x direction excluding ghost cells
    const Domain &domain() const;

    /// access cell size in x-direction
    double dx() const;
    /// access cell size in y-direction
    double dy() const;

    /**
     * @brief Access inflow cells
     *
     * @param[out] vector of fluid cells
     */
    const std::vector<Cell *> &fluid_cells() const;

    /**
     * @brief Access moving wall cells
     *
     * @param[out] vector of moving wall cells
     */
    const std::vector<Cell *> &moving_wall_cells() const;

    /**
     * @brief Access fixed wall cells
     *
     * @param[out] vector of fixed wall cells
     */
    const std::vector<Cell *> &fixed_wall_cells() const;

    /**
     * @brief Access fixed wall cells
     *
     * @param[out] vector of fixed heat wall cells
     */
    const std::vector<Cell *> &fixed_wall_heat_cells() const;

    /**
     * @brief Access fixed wall cold cells
     *
     * @param[out] vector of fixed wall cells
     */
    const std::vector<Cell *> &fixed_wall_cold_cells() const;

    /**
     * @brief Access fixed wall cells
     *
     * @param[out] vector of fixed wall cells
     */
    const std::vector<Cell *> &inflow_wall_cells() const;

    /**
     * @brief Access fixed wall cells
     *
     * @param[out] vector of fixed wall cells
     */
    const std::vector<Cell *> &outflow_wall_cells() const;

    /**
     * @brief getter method for useTemp variable
     *
     */
    bool getUseTemp();

    /**
     * @brief getter method for useTemp variable
     *
     * @param[in] bool if temperature is used
     */
    void setUseTemp(bool useTemp);

  private:
    /**@brief Default lid driven cavity case generator
     *
     * This function creates default lid driven cavity
     * case without need for a pgm file
     */
    void build_lid_driven_cavity();

    /// Build cell data structures with given geometrical data
    void assign_cell_types(std::vector<std::vector<int>> &geometry_data);
    /// Extract geometry from pgm file and create geometrical data
    void parse_geometry_file(std::string filedoc, std::vector<std::vector<int>> &geometry_data);

    /// Actual matrix of all cells (including ghost cells)
    Matrix<Cell> _cells;
    /// Vector of pointers to all fluid cells
    std::vector<Cell *> _fluid_cells;
    /// Vector of pointers to all cells belonging to fixed walls
    std::vector<Cell *> _fixed_wall_cells;
    /// Vector of pointers to all cells belonging to fixed walls heat
    std::vector<Cell *> _fixed_wall_heat_cells;
    /// Vector of pointers to all cells belonging to fixed walls cold
    std::vector<Cell *> _fixed_wall_cold_cells;
    /// Vector of pointers to all cells belonging to moving walls
    std::vector<Cell *> _moving_wall_cells;
    /// Vector of pointers to all cells belonging to inflow walls
    std::vector<Cell *> _inflow_wall_cells;
    /// Vector of pointers to all cells belonging to outflow walls
    std::vector<Cell *> _outflow_wall_cells;

    /// flag to know if temperature calculation should be performed
    bool _useTemp;

    /// Domain object holding geometrical information
    Domain _domain;

    int _rank;
    int _size;
    int _iProc;
    int _jProc;
    int _xCoord;
    int _yCoord;
};

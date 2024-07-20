#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Boundary.hpp"
#include "Discretization.hpp"
#include "Domain.hpp"
#include "Fields.hpp"
#include "Grid.hpp"
#include "PressureSolver.hpp"
#include "Health.h"
#include "Communication.hpp"

/**
 * @brief Class to hold and orchestrate the simulation flow.
 *
 */
class Case {
  public:
    /**
     * @brief Parallel constructor for the Case.
     *
     * Reads input file, creates Fields, Grid, Boundary, Solver and sets
     * Discretization parameters. Creates output directory.
     *
     * @param[in] Input file name
     */
    Case(std::string file_name, int argn, char **args, int rank, int size);

    /**
     * @brief Main function to simulate the flow until the end time.
     *
     * Calculates the fluxes
     * Calculates the right hand side
     * Solves pressure
     * Calculates velocities
     * Outputs the solution files
     */
    void simulate(std::ofstream& logging);

  private:
    /// Plain case name without paths
    std::string _case_name;
    /// Output directory name
    std::string _dict_name;
    /// Geometry file name
    std::string _geom_name{"NONE"};
    /// Relative input file path
    std::string _prefix;

    /// Simulation time
    double _t_end;
    /// Solution file outputting frequency
    double _output_freq;

    Fields _field;
    Grid _grid;
    Discretization _discretization;
    std::unique_ptr<PressureSolver> _pressure_solver;
    /// Collection of all boundaries for the simulated case
    std::vector<std::unique_ptr<Boundary>> _boundaries;

    /// Solver convergence tolerance
    double _tolerance;

    /// Maximum number of iterations for the solver
    int _max_iter;

    /// Logic to calculate health of the simulation
    Health _health;

    /// flag to check if temperature calculation is done or not
    bool _useTemp;

    /**
     * @brief Creating file names from given input data file
     *
     * Extracts path of the case file and creates code-readable file names
     * for outputting directory and geometry file.
     *
     * @param[in] input data file
     */

    int _rank;
    int _size;
    int _iProc;
    int _jProc;

    void set_file_names(std::string file_name);

    /**
     * @brief Solution file outputter
     *
     * Outputs the solution files in .vtk format. Ghost cells are excluded.
     * Pressure is cell variable while velocity is point variable while being
     * interpolated to the cell faces
     *
     * @param[in] Timestep of the solution
     * @param[in] Current rank of the executing process
     */
    void output_vtk(int t);

    /**
     * @brief Fill out domain object
     *
     * Fills the Domain object with the geometrical information about the domain size. In case of running the code in
     * parallel, the information should correspond to the subdomain belonging to the executing MPI-process after
     * decomposition.
     *
     * @param[in] Reference to the domain object
     * @param[in] Number of cells in x-direction for this MPI rank
     * @param[in] Number of cells in y-direction for this MPI rank
     */
    void build_domain(Domain &domain, int imax_domain, int jmax_domain);
};

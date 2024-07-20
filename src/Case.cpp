#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace filesystem = std::filesystem;

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridWriter.h>
#include <vtkTuple.h>

#include "Case.hpp"
#include "Enums.hpp"

Case::Case(std::string file_name, int argn, char **args, int rank, int size) {

    _rank = rank;
    _size = size;
    // Read input parameters
    const int MAX_LINE_LENGTH = 1024;
    std::ifstream file(file_name);
    double nu{};      /* viscosity   */
    double UI{};      /* velocity x-direction */
    double VI{};      /* velocity y-direction */
    double PI{};      /* pressure */
    double GX{};      /* gravitation x-direction */
    double GY{};      /* gravitation y-direction */
    double xlength{}; /* length of the domain x-dir.*/
    double ylength{}; /* length of the domain y-dir.*/
    double dt{};      /* time step */
    int imax{};       /* number of cells x-direction*/
    int jmax{};       /* number of cells y-direction*/
    double gamma{};   /* uppwind differencing factor*/
    double omg{};     /* relaxation factor */
    double tau{};     /* safety factor for time step*/
    int itermax{};    /* max. number of iterations for pressure per time step */
    double eps{};     /* accuracy bound for pressure*/
    double UIN{0.0};  /* inlet velocity x direction*/
    double VIN{0.0};  /* inlet velocity y direction*/
    double POut{0.0}; /* outlet pressure*/
    double TI{0.0};   /* initial temperature*/
    double TH{0.0};   /* hot wall temperature*/
    double TC{0.0};   /* cold wall temperature*/
    double alpha{0.0};/* thermal diffusivity*/
    double beta{0.0}; /* thermal expansion*/





    if (file.is_open()) {

        std::string var;
        while (!file.eof() && file.good()) {
            file >> var;
            if (var[0] == '#') { /* ignore comment line*/
                file.ignore(MAX_LINE_LENGTH, '\n');
            } else {
                if (var == "xlength") file >> xlength;
                if (var == "ylength") file >> ylength;
                if (var == "nu") file >> nu;
                if (var == "t_end") file >> _t_end;
                if (var == "dt") file >> dt;
                if (var == "omg") file >> omg;
                if (var == "eps") file >> eps;
                if (var == "tau") file >> tau;
                if (var == "gamma") file >> gamma;
                if (var == "dt_value") file >> _output_freq;
                if (var == "UI") file >> UI;
                if (var == "VI") file >> VI;
                if (var == "GX") file >> GX;
                if (var == "GY") file >> GY;
                if (var == "PI") file >> PI;
                if (var == "itermax") file >> itermax;
                if (var == "imax") file >> imax;
                if (var == "jmax") file >> jmax;
                if (var == "geo_file") file >> _geom_name;
                if (var == "UIN") file >> UIN;
                if (var == "VIN") file >> VIN;
                if (var == "POut") file >> POut;
                if (var == "TI") file >> TI;
                if (var == "wall_temp_4") file >> TH;
                if (var == "wall_temp_5") file >> TC;
                if (var == "alpha") file >> alpha;
                if (var == "beta") file >> beta;
                if (var == "useTemp") {
                    std::string tmp{};
                    file >> tmp;
                    if  (tmp == "true") {
                        _useTemp = true;
                    }
                }
                if (var == "iProc") {
                    file >> _iProc;
                    if ( _iProc < 1 ) {
                        Communication::finalize();
                        exit(0);
                    }
                }
                if (var == "jProc") {
                    file >> _jProc;
                    if ( _iProc < 1 ) {
                        Communication::finalize();
                        exit(0);
                    }
                }
            }
        }
    }
    file.close();

    if ((_iProc * _jProc) != _size) {
        Communication::finalize();
        exit(0);
    }

/*
    int dims[2] = {_iProc, _jProc};
    MPI_Dims_create(_size, 2, dims);

    int periods[2] = {false, false};
    int reorder = false;

    // Create a communicator with a cartesian topology.
    MPI_Comm new_communicator;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &new_communicator);

    // Get my coordinates in the new communicator
    int my_coords[2];
    MPI_Cart_coords(new_communicator, _rank, 2, my_coords);

    // Declare our neighbours
    enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};
    char* neighbours_names[4] = {"down", "up", "left", "right"};
    int neighbours_ranks[4];

    // Let consider dim[0] = X, so the shift tells us our left and right neighbours
    MPI_Cart_shift(new_communicator, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);

    // Let consider dim[1] = Y, so the shift tells us our up and down neighbours
    MPI_Cart_shift(new_communicator, 1, 1, &neighbours_ranks[DOWN], &neighbours_ranks[UP]);

    MPI_Comm_rank(new_communicator, &_rank);
*/

    std::map<int, double> wall_vel;
    if (_geom_name.compare("NONE") == 0) {
        wall_vel.insert(std::pair<int, double>(LidDrivenCavity::moving_wall_id, LidDrivenCavity::wall_velocity));
    }

    // Set file names for geometry file and output directory
    set_file_names(file_name);


    // Build up the domain
    Domain domain;
    domain.dx = xlength / static_cast<double>(imax);
    domain.dy = ylength / static_cast<double>(jmax);
    domain.domain_imax = imax;
    domain.domain_jmax = jmax;

    build_domain(domain, imax, jmax);

    _grid = Grid(_geom_name, domain, _iProc, _jProc, _rank, _size);

    _grid.setUseTemp(_useTemp);
    _field = Fields(_grid, nu, dt, tau, alpha, beta, UI, VI, PI, TI, GX, GY);

    _discretization = Discretization(domain.dx, domain.dy, gamma);
    _pressure_solver = std::make_unique<SOR>(omg);
    _max_iter = itermax;
    _tolerance = eps;

    // we need different maps as we cannot access the
    std::map<int, double> wall_temp_adiabatic{{3, -1}};
    std::map<int, double> wall_temp_heat{{4, TH}};
    std::map<int, double> wall_temp_cold{{5, TC}};


    // Construct boundaries
    if (not _grid.moving_wall_cells().empty()) {
        _boundaries.push_back(
            std::make_unique<MovingWallBoundary>(_grid.moving_wall_cells(), LidDrivenCavity::wall_velocity));
    }
    if (not _grid.fixed_wall_cells().empty()) {
        _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_cells(), wall_temp_adiabatic));
    }
    if (not _grid.fixed_wall_heat_cells().empty()) {
        _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_heat_cells(), wall_temp_heat));
    }
    if (not _grid.fixed_wall_cold_cells().empty()) {
        _boundaries.push_back(std::make_unique<FixedWallBoundary>(_grid.fixed_wall_cold_cells(), wall_temp_cold));
    }
    if (not _grid.inflow_wall_cells().empty()) {
        _boundaries.push_back(std::make_unique<InflowBoundary>(_grid.inflow_wall_cells(), UIN, VIN));
    }
    if (not _grid.inflow_wall_cells().empty()) {
        _boundaries.push_back(std::make_unique<OutflowBoundary>(_grid.outflow_wall_cells(), POut));
    }
}

void Case::set_file_names(std::string file_name) {
    std::string temp_dir;
    bool case_name_flag = true;
    bool prefix_flag = false;

    for (int i = file_name.size() - 1; i > -1; --i) {
        if (file_name[i] == '/') {
            case_name_flag = false;
            prefix_flag = true;
        }
        if (case_name_flag) {
            _case_name.push_back(file_name[i]);
        }
        if (prefix_flag) {
            _prefix.push_back(file_name[i]);
        }
    }

    for (int i = file_name.size() - _case_name.size() - 1; i > -1; --i) {
        temp_dir.push_back(file_name[i]);
    }

    std::reverse(_case_name.begin(), _case_name.end());
    std::reverse(_prefix.begin(), _prefix.end());
    std::reverse(temp_dir.begin(), temp_dir.end());

    _case_name.erase(_case_name.size() - 4);
    _dict_name = temp_dir;
    _dict_name.append(_case_name);
    _dict_name.append("_Output");

    if (_geom_name.compare("NONE") != 0) {
        _geom_name = _prefix + _geom_name;
    }

    // Create output directory
    filesystem::path folder(_dict_name);
    try {
        filesystem::create_directory(folder);
    } catch (const std::exception &e) {
        std::cerr << "Output directory could not be created." << std::endl;
        std::cerr << "Make sure that you have write permissions to the "
                     "corresponding location"
                  << std::endl;
    }
}



/**
 * This function is the main simulation loop. In the simulation loop, following steps are required
 * - Calculate and apply velocity boundary conditions for all the boundaries in _boundaries container
 *   using applyVelocity() member function of Boundary class
 * - Calculate fluxes (F and G) using calculate_fluxes() member function of Fields class.
 *   Flux consists of diffusion and convection part, which are located in Discretization class
 * - Apply Flux boundary conditions using applyFlux()
 * - Calculate right-hand-side of PPE using calculate_rs() member function of Fields class
 * - Iterate the pressure poisson equation until the residual becomes smaller than the desired tolerance
 *   or the maximum number of the iterations are performed using solve() member function of PressureSolver
 * - Update pressure boundary conditions after each iteration of the SOR solver
 * - Calculate the velocities u and v using calculate_velocities() member function of Fields class
 * - calculate the maximal timestep size for the next iteration using calculate_dt() member function of Fields class
 * - Write vtk files using output_vtk() function
 *
 * Please note that some classes such as PressureSolver, Boundary are abstract classes which means they only provide the
 * interface and/or common functions. You need to define functions with individual functionalities in inherited
 * classes such as MovingWallBoundary class.
 *
 * For information about the classes and functions, you can check the header files.
 */
void Case::simulate(std::ofstream& logging) {


    if (_rank == 0) {
        std::cout << "Starting simulation with " << _size << " processes" << std::endl;
        logging << "Starting simulation with " << _size << " processes \n";
    }


    double t = 0.0;
    double dt = _field.dt();
    int timestep = 0;
    double output_counter = 0.0;
    double avg_iter{0.0};
    int not_conv{0};
    int current_iteration = 0;

    int num_fluid_cells;


    output_vtk(timestep++);

    while (t < _t_end) {

        // applyVelocity and applyTemperature boundary conditions
        for (auto &j : _boundaries) {
            j->applyVelocity(_field);
            if (_useTemp) {
                j->applyTemperature(_field);
            }
        }

        if (_useTemp) {
            // calculate temperature values
            _field.calculate_temperatures(_grid);
            Communication::communicate(_field.t_matrix(), _grid.domain(), _rank);
        }

        // calculate fluxes using member function of Fields class
        _field.calculate_fluxes(_grid);

        // apply boundary condition for F and G (fluxes)
        for (auto &j : _boundaries) {
            j->applyFlux(_field);
        }

        Communication::communicate(_field.f_matrix(), _grid.domain(), _rank);
        Communication::communicate(_field.g_matrix(), _grid.domain(), _rank);

        // RHS of Pressure Poisson Equation
        _field.calculate_rs(_grid);

        // Iterate Pressure Poisson Equation; 
        int it{0};
        double res{1000.};
        while (it < _max_iter && res >= _tolerance) {
            for (auto &j : _boundaries) {
                j->applyPressure(_field);
            }
            // DONE: Applying it here ensures that the linear system for the pressure values has a (physical) solution
            res = _pressure_solver->solve(_field, _grid);
            it++;

            res = Communication::reduce_sum(res);

            num_fluid_cells = _grid.fluid_cells().size();
            num_fluid_cells = Communication::reduce_sum(num_fluid_cells);
            
            res = res/num_fluid_cells; 

            res=std::sqrt(res);  

            Communication::communicate(_field.p_matrix(), _grid.domain(), _rank);
        }

        if (res >= _tolerance) {
            not_conv++;
        }
        avg_iter += it;


        _field.calculate_velocities(_grid);
        Communication::communicate(_field.u_matrix(), _grid.domain(), _rank);
        Communication::communicate(_field.v_matrix(), _grid.domain(), _rank);

        output_counter += dt;

        if (output_counter >= _output_freq) {
            output_vtk(timestep++);
            output_counter = 0;
        }
        if (_rank == 0) {
            if (current_iteration % 100 == 0) {
            std::string isHealthy =  _health.isHealthy(_field, _grid) ? "yes" : "no";
            std::cout << "iteration: " << current_iteration << " | t: " << t << " | step-size: " << dt << " | healthy: " << isHealthy <<  std::endl;
            logging << "iteration: " << current_iteration << " | t: " << t << " | step-size: " << dt << " | healthy: " << isHealthy << "\n";
        }
        }
        

        t+= dt;
        current_iteration++;

        dt = _field.calculate_dt(_grid);
        dt = Communication::reduce_min(dt);
    }

    if (_rank == 0) {
        avg_iter /= current_iteration;
        std::cout << "av_iterations: " << avg_iter << ", number not converged: " << not_conv << std::endl;
        logging << "av_iterations: " << avg_iter << ", number not converged: " << not_conv << std::endl;
    }

    output_vtk(timestep);
}

void Case::output_vtk(int timestep) {
    // Create a new structured grid
    vtkSmartPointer<vtkStructuredGrid> structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();

    // Create grid
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    double dx = _grid.dx();
    double dy = _grid.dy();

    double x = _grid.domain().iminb * dx;
    double y = _grid.domain().jminb * dy;

    { y += dy; }
    { x += dx; }

    double z = 0;
    for (int col = 0; col < _grid.domain().size_y + 1; col++) {
        x = _grid.domain().iminb * dx;
        { x += dx; }
        for (int row = 0; row < _grid.domain().size_x + 1; row++) {
            points->InsertNextPoint(x, y, z);
            x += dx;
        }
        y += dy;
    }

    // Specify the dimensions of the grid
    structuredGrid->SetDimensions(_grid.domain().size_x + 1, _grid.domain().size_y + 1, 1);
    structuredGrid->SetPoints(points);

    

    std::vector<vtkIdType> fixed_wall_cells;
    for (int i = 1; i <= _grid.size_x(); i++) {
        for (int j = 1; j <= _grid.size_y(); j++) {
            if (_grid.cell(i, j).wall_id() != 0) {
                fixed_wall_cells.push_back(i - 1 + (j - 1) * _grid.size_x());
            }
        }
    }

    for (auto t = 0; t < fixed_wall_cells.size(); t++) {
        structuredGrid->BlankCell(fixed_wall_cells.at(t));
    }

    // Pressure Array
    vtkSmartPointer<vtkDoubleArray> Pressure = vtkSmartPointer<vtkDoubleArray>::New();
    Pressure->SetName("pressure");
    Pressure->SetNumberOfComponents(1);

    // Velocity Array for cell data
    vtkSmartPointer<vtkDoubleArray> Velocity = vtkSmartPointer<vtkDoubleArray>::New();
    Velocity->SetName("velocity");
    Velocity->SetNumberOfComponents(3);

    // Temperature Array
    vtkSmartPointer<vtkDoubleArray> Temperature = vtkSmartPointer<vtkDoubleArray>::New();
    Temperature->SetName("temperature");
    Temperature->SetNumberOfComponents(1);

    // Temp Velocity
    float vel[3];
    vel[2] = 0; // Set z component to 0

    // Print pressure, velocity and temperature from bottom to top
    for (int j = 1; j < _grid.domain().size_y + 1; j++) {
        for (int i = 1; i < _grid.domain().size_x + 1; i++) {
            double pressure = _field.p(i, j);
            Pressure->InsertNextTuple(&pressure);
            vel[0] = (_field.u(i - 1, j) + _field.u(i, j)) * 0.5;
            vel[1] = (_field.v(i, j - 1) + _field.v(i, j)) * 0.5;
            Velocity->InsertNextTuple(vel);
            if (_useTemp) {
                double temperature{_field.t(i,j)};
                Temperature->InsertNextTuple(&temperature);
            }
        }
    }

    // Velocity Array for point data
    vtkSmartPointer<vtkDoubleArray> VelocityPoints = vtkSmartPointer<vtkDoubleArray>::New();
    VelocityPoints->SetName("velocity");
    VelocityPoints->SetNumberOfComponents(3);

    // Print Velocity from bottom to top
    for (int j = 0; j < _grid.domain().size_y + 1; j++) {
        for (int i = 0; i < _grid.domain().size_x + 1; i++) {
            vel[0] = (_field.u(i, j) + _field.u(i, j + 1)) * 0.5;
            vel[1] = (_field.v(i, j) + _field.v(i + 1, j)) * 0.5;
            VelocityPoints->InsertNextTuple(vel);
        }
    }

    // Add Pressure to Structured Grid
    structuredGrid->GetCellData()->AddArray(Pressure);

    // Add Velocity to Structured Grid
    structuredGrid->GetCellData()->AddArray(Velocity);
    structuredGrid->GetPointData()->AddArray(VelocityPoints);

    // Add Temperature to Structured Grid
    if (_useTemp) {
        structuredGrid->GetCellData()->AddArray(Temperature);
    }

    // Write Grid
    vtkSmartPointer<vtkStructuredGridWriter> writer = vtkSmartPointer<vtkStructuredGridWriter>::New();

    // Create Filename
    std::string outputname =
        _dict_name + '/' + _case_name + "_" + std::to_string(_rank) + "." + std::to_string(timestep) + ".vtk";

    writer->SetFileName(outputname.c_str());
    writer->SetInputData(structuredGrid);
    writer->Write();
}

void Case::build_domain(Domain &domain, int imax_domain, int jmax_domain) {



    int I, J;

    int imin, imax, jmin, jmax, size_x, size_y;

    if (_rank == 0) {
        // std::cout << domain.size_x << " in case " << domain.size_y << std::endl;
        for (int i = 1; i < _size; ++i) {
            I = i % _iProc + 1;
            J = i / _iProc + 1;
            imin = (I - 1) * (imax_domain / _iProc);
            imax = I * (imax_domain / _iProc) + 2;
            jmin = (J - 1) * (jmax_domain / _jProc);
            jmax = J * (jmax_domain / _jProc) + 2;
            size_x = imax_domain / _iProc;
            size_y = jmax_domain / _jProc;

            // Adding the extra cells when number of cells is not divisible by iproc and jproc
            if (I == _iProc) {
                imax = imax_domain + 2;
                size_x = imax - imin - 2;
            }

            if (J == _jProc) {
                jmax = jmax_domain + 2;
                size_y = jmax - jmin - 2;
            }

            std::array<int, 4> neighbours = {-1, -1, -1, -1};
            if (I > 1) {
                // Left neighbour
                neighbours[0] = i - 1;
            }

            if (J > 1) {
                // Bottom neighbour
                neighbours[3] = i - _iProc;
            }

            if (_iProc > 1 && I < (_size - 1) % _iProc + 1) {
                // Right neighbour
                neighbours[1] = i + 1;
            }

            if (_jProc > 1 && J < (_size - 1) / _iProc + 1) {
                // Top neighbour
                neighbours[2] = i + _iProc;
            }

            MPI_Send(&imin, 1, MPI_INT, i, 999, MPI_COMM_WORLD);
            MPI_Send(&imax, 1, MPI_INT, i, 998, MPI_COMM_WORLD);
            MPI_Send(&jmin, 1, MPI_INT, i, 997, MPI_COMM_WORLD);
            MPI_Send(&jmax, 1, MPI_INT, i, 996, MPI_COMM_WORLD);
            MPI_Send(&size_x, 1, MPI_INT, i, 995, MPI_COMM_WORLD);
            MPI_Send(&size_y, 1, MPI_INT, i, 994, MPI_COMM_WORLD);
            MPI_Send(neighbours.data(), 4, MPI_INT, i, 993, MPI_COMM_WORLD);
        }
        // For rank 0
        I = _rank % _iProc + 1;
        J = _rank / _iProc + 1;
        domain.iminb = (I - 1) * (imax_domain / _iProc);
        domain.imaxb = I * (imax_domain / _iProc) + 2;
        domain.jminb = (J - 1) * (jmax_domain / _jProc);
        domain.jmaxb = J * (jmax_domain / _jProc) + 2;
        domain.size_x = imax_domain / _iProc;
        domain.size_y = jmax_domain / _jProc;
        domain.neighbours[0] = -1; // left
        domain.neighbours[1] = -1; // right
        if (_iProc > 1) domain.neighbours[1] = 1;
        domain.neighbours[2] = -1; // top
        if (_jProc > 1) domain.neighbours[2] = _iProc;
        domain.neighbours[3] = -1; // bottom
    } else {
        MPI_Status status;
        MPI_Recv(&domain.iminb, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, &status);
        MPI_Recv(&domain.imaxb, 1, MPI_INT, 0, 998, MPI_COMM_WORLD, &status);
        MPI_Recv(&domain.jminb, 1, MPI_INT, 0, 997, MPI_COMM_WORLD, &status);
        MPI_Recv(&domain.jmaxb, 1, MPI_INT, 0, 996, MPI_COMM_WORLD, &status);
        MPI_Recv(&domain.size_x, 1, MPI_INT, 0, 995, MPI_COMM_WORLD, &status);
        MPI_Recv(&domain.size_y, 1, MPI_INT, 0, 994, MPI_COMM_WORLD, &status);
        MPI_Recv(&domain.neighbours, 4, MPI_INT, 0, 993, MPI_COMM_WORLD, &status);
    }
}


    /*
    domain.iminb = x_coord * (imax_domain / _iProc);
    domain.imaxb = (x_coord + 1) * (imax_domain / _iProc) + 2;
    domain.jminb = y_coord * (jmax_domain / _jProc);
    // TODO: changed from jminb to jmaxb?
    domain.jmaxb = (y_coord + 1) * (jmax_domain / _jProc) + 2;
    domain.size_x = imax_domain / _iProc;
    domain.size_y = jmax_domain / _jProc;

    if ( (x_coord + 1) == _iProc ) {
        domain.imaxb = imax_domain + 2;
        domain.size_x = domain.imaxb - domain.iminb - 2;  
    }

    if ( (y_coord + 1) == _jProc ) {
        domain.imaxb = jmax_domain + 2;
        domain.size_y = domain.jmaxb - domain.jminb - 2;  
    }

    MyFile << "In build domain. my rank: " << _rank;
    MyFile << " iminb: " << domain.iminb << ", imaxb: " << domain.imaxb << ", jminb: " << domain.jminb <<
        ", jmaxb: " << domain.jmaxb << ", size x: " << domain.size_x << ", size y: " << domain.size_y << std::endl;

    // domain.iminb = 0;
    // domain.jminb = 0;
    // domain.imaxb = imax_domain + 2;
    // domain.jmaxb = jmax_domain + 2;
    // domain.size_x = imax_domain;
    // domain.size_y = jmax_domain;

    */



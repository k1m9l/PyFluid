#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Enums.hpp"
#include "Grid.hpp"

Grid::Grid(std::string geom_name, Domain &domain, int iProc, int jProc, int rank, int size) {

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    _domain = domain;
    _iProc = iProc;
    _jProc = jProc;
    _rank = rank;
    _size = size;

    _cells = Matrix<Cell>(_domain.size_x + 2, _domain.size_y + 2);

    // geometry_data is a subset of the full geometry data
    if (geom_name.compare("NONE")) {
        std::vector<std::vector<int>> geometry_data(_domain.size_x + 2,
                                                    std::vector<int>(_domain.size_y + 2, 0));
        parse_geometry_file(geom_name, geometry_data);
        assign_cell_types(geometry_data);
    } else {
        build_lid_driven_cavity();
    }
}

void Grid::build_lid_driven_cavity() {
    std::vector<std::vector<int>> geometry_data(_domain.size_x + 2, std::vector<int>(_domain.size_y + 2, 0));

    for (int i = 0; i < _domain.size_x + 2; ++i) {
        for (int j = 0; j < _domain.size_y + 2; ++j) {
            // Bottom, left and right walls: no-slip
            if ((i == 0 && _domain.iminb == 0) ||
                (i == _domain.size_x + 1 && _domain.imaxb == _domain.domain_imax + 2) ||
                (j == 0 && _domain.jminb == 0)) {
                geometry_data.at(i).at(j) = LidDrivenCavity::fixed_wall_id;
            }
            // Top wall: moving wall
            else if (j == _domain.size_y + 1 && _domain.jmaxb == _domain.domain_jmax + 2) {
                geometry_data.at(i).at(j) = LidDrivenCavity::moving_wall_id;
            }
        }
    }
    assign_cell_types(geometry_data);
}

void Grid::assign_cell_types(std::vector<std::vector<int>> &geometry_data) {

    int i = 0;
    int j = 0;

    for (int j_geom = 0; j_geom < _domain.size_y + 2; ++j_geom) {
        { i = 0; }
        for (int i_geom = 0; i_geom < _domain.size_x + 2; ++i_geom) {

            bool forParallelization = false;

            if (i_geom == 0 && _domain.neighbours[0] != -1) {
                forParallelization = true;
            } else if (i_geom == (_domain.size_x + 1) && _domain.neighbours[1] != -1) {
                forParallelization = true;
            } else if (j_geom == (_domain.size_y + 1) && _domain.neighbours[2] != -1) {
                forParallelization = true;
            } else if (j_geom == 0 && _domain.neighbours[3] != -1) {
                forParallelization = true;
            }

            if (!forParallelization) {
                if (geometry_data.at(i_geom).at(j_geom) == 0) {
                    _cells(i, j) = Cell(i, j, cell_type::FLUID);
                    _fluid_cells.push_back(&_cells(i, j));
                } else if (geometry_data.at(i_geom).at(j_geom) == LidDrivenCavity::moving_wall_id) {
                    _cells(i, j) = Cell(i, j, cell_type::MOVING_WALL, geometry_data.at(i_geom).at(j_geom));
                    _moving_wall_cells.push_back(&_cells(i, j));
                } else if (geometry_data.at(i_geom).at(j_geom) == 1) {
                    _cells(i, j) = Cell(i, j, cell_type::INFLOW, geometry_data.at(i_geom).at(j_geom));
                    _inflow_wall_cells.push_back(&_cells(i, j));
                } else if (geometry_data.at(i_geom).at(j_geom) == 2) {
                    _cells(i, j) = Cell(i, j, cell_type::OUTFLOW, geometry_data.at(i_geom).at(j_geom));
                    _outflow_wall_cells.push_back(&_cells(i, j));
                }
                else if (geometry_data.at(i_geom).at(j_geom) == 4) {
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL_HEAT, geometry_data.at(i_geom).at(j_geom));
                    _fixed_wall_heat_cells.push_back(&_cells(i, j));
                }
                else if (geometry_data.at(i_geom).at(j_geom) == 5) {
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL_COLD, geometry_data.at(i_geom).at(j_geom));
                    _fixed_wall_cold_cells.push_back(&_cells(i, j));
                }
                else {
                    // Outer walls and inner obstacles
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                    _fixed_wall_cells.push_back(&_cells(i, j));
                }
            } 
            else {

                if (geometry_data.at(i_geom).at(j_geom) == 0) {
                    _cells(i, j) = Cell(i, j, cell_type::FLUID);
                } else if (geometry_data.at(i_geom).at(j_geom) == LidDrivenCavity::moving_wall_id) {
                    _cells(i, j) = Cell(i, j, cell_type::MOVING_WALL, geometry_data.at(i_geom).at(j_geom));
                } else if (geometry_data.at(i_geom).at(j_geom) == 1) {
                    _cells(i, j) = Cell(i, j, cell_type::INFLOW, geometry_data.at(i_geom).at(j_geom));
                } else if (geometry_data.at(i_geom).at(j_geom) == 2) {
                    _cells(i, j) = Cell(i, j, cell_type::OUTFLOW, geometry_data.at(i_geom).at(j_geom));
                }
                else if (geometry_data.at(i_geom).at(j_geom) == 4) {
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL_HEAT, geometry_data.at(i_geom).at(j_geom));
                }
                else if (geometry_data.at(i_geom).at(j_geom) == 5) {
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL_COLD, geometry_data.at(i_geom).at(j_geom));
                }
                else {
                    // Outer walls and inner obstacles
                    _cells(i, j) = Cell(i, j, cell_type::FIXED_WALL, geometry_data.at(i_geom).at(j_geom));
                }

            }

            ++i;
        }
        ++j;
    }

    // Corner cell neighbour assigment
    // Bottom-Left Corner
    i = 0;
    j = 0;
    _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
    _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
    if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::TOP);
    }
    if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::RIGHT);
    }
    // Top-Left Corner
    i = 0;
    j = _domain.size_y + 1;
    _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
    _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
    if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::BOTTOM);
    }
    if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::RIGHT);
    }

    // Top-Right Corner
    i = _domain.size_x + 1;
    j = Grid::_domain.size_y + 1;
    _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
    _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
    if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::BOTTOM);
    }
    if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::LEFT);
    }

    // Bottom-Right Corner
    i = Grid::_domain.size_x + 1;
    j = 0;
    _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
    _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
    if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::TOP);
    }
    if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
        _cells(i, j).add_border(border_position::LEFT);
    }
    // Bottom cells
    j = 0;
    for (int i = 1; i < _domain.size_x + 1; ++i) {
        _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
        _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
        _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
        if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::RIGHT);
        }
        if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::LEFT);
        }
        if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::TOP);
        }
    }

    // Top Cells
    j = Grid::_domain.size_y + 1;

    for (int i = 1; i < _domain.size_x + 1; ++i) {
        _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
        _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
        _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
        if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::RIGHT);
        }
        if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::LEFT);
        }
        if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::BOTTOM);
        }
    }

    // Left Cells
    i = 0;
    for (int j = 1; j < _domain.size_y + 1; ++j) {
        _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
        _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
        _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
        if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::RIGHT);
        }
        if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::BOTTOM);
        }
        if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::TOP);
        }
    }
    // Right Cells
    i = Grid::_domain.size_x + 1;
    for (int j = 1; j < _domain.size_y + 1; ++j) {
        _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
        _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);
        _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
        if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::LEFT);
        }
        if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::BOTTOM);
        }
        if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
            _cells(i, j).add_border(border_position::TOP);
        }
    }

    // Inner cells
    for (int i = 1; i < _domain.size_x + 1; ++i) {
        for (int j = 1; j < _domain.size_y + 1; ++j) {
            _cells(i, j).set_neighbour(&_cells(i + 1, j), border_position::RIGHT);
            _cells(i, j).set_neighbour(&_cells(i - 1, j), border_position::LEFT);
            _cells(i, j).set_neighbour(&_cells(i, j + 1), border_position::TOP);
            _cells(i, j).set_neighbour(&_cells(i, j - 1), border_position::BOTTOM);

            if (_cells(i, j).type() != cell_type::FLUID) {
                if (_cells(i, j).neighbour(border_position::LEFT)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::LEFT);
                }
                if (_cells(i, j).neighbour(border_position::RIGHT)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::RIGHT);
                }
                if (_cells(i, j).neighbour(border_position::BOTTOM)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::BOTTOM);
                }
                if (_cells(i, j).neighbour(border_position::TOP)->type() == cell_type::FLUID) {
                    _cells(i, j).add_border(border_position::TOP);
                }
            }
        }
    }
}

void Grid::parse_geometry_file(std::string filedoc, std::vector<std::vector<int>> &geometry_data) {
    if (_rank == 0) {
    
        int num_cells_in_x, num_cells_in_y, depth;

        // Full geometry data is read
        std::vector<std::vector<int>> full_geometry_data(_domain.domain_imax + 2, std::vector<int>(_domain.domain_jmax + 2, 0) );

        std::ifstream infile(filedoc);
        std::stringstream ss;
        std::string inputLine = "";

        // First line : version
        getline(infile, inputLine);
        if (inputLine.compare("P2") != 0) {
            std::cerr << "First line of the PGM file should be P2" << std::endl;
        }

        // Second line : comment
        getline(infile, inputLine);

        // Continue with a stringstream
        ss << infile.rdbuf();
        // Third line : size
        ss >> num_cells_in_x >> num_cells_in_y;
        // Fourth line : depth
        ss >> depth;

        // Following lines : data (origin of x-y coordinate system in bottom-left corner)
        for (int y = num_cells_in_y - 1; y > -1; --y) {
            for (int x = 0; x < num_cells_in_x; ++x) {
                ss >> full_geometry_data[x][y];
            }
        }

        infile.close();

        int I, J;
        int imin, jmin, imax, jmax;

        for (int i = 1; i < _size; ++i) {
            I = i % _iProc + 1;
            J = i / _iProc + 1;
            imin = (I-1) * ((num_cells_in_x - 2) / _iProc);
            imax = I * ((num_cells_in_x - 2) / _iProc) + 2;
            jmin = (J-1) * ((num_cells_in_y - 2) / _jProc);
            jmax = J * ((num_cells_in_y - 2) / _jProc) + 2;

            if (I == _iProc) imax = num_cells_in_x;
            if (J == _jProc) jmax = num_cells_in_y;

            std::vector<int> rank_geometry_data;
            for (int x = imin; x < imax; ++x) {
                for (int y = jmin; y < jmax; ++y) {
                    rank_geometry_data.push_back(full_geometry_data[x][y]);
                }
            }

            MPI_Send(rank_geometry_data.data(), rank_geometry_data.size(), MPI_INT, i, 999999, MPI_COMM_WORLD);
        }

        for (int y = 0; y < _domain.size_y + 2; ++y) {
            for (int x = 0; x < _domain.size_x + 2; ++x) {
                geometry_data[x][y] = full_geometry_data[x][y];
            }
        }
    } 
    else {
        std::vector<int> rank_geometry_data((_domain.size_x + 2) * (_domain.size_y + 2), 0);

        MPI_Status status;

        MPI_Recv(rank_geometry_data.data(), rank_geometry_data.size(), MPI_INT, 0, 999999, MPI_COMM_WORLD, &status);

        for (int y = 0; y < _domain.size_y + 2; ++y) {
            for (int x = 0; x < _domain.size_x + 2; ++x) {
                geometry_data.at(x).at(y) = rank_geometry_data[x * (_domain.size_y + 2) + y];
            }
        }
    }

}

int Grid::size_x() const { return _domain.size_x; }
int Grid::size_y() const { return _domain.size_y; }

Cell Grid::cell(int i, int j) const { return _cells(i, j); }

double Grid::dx() const { return _domain.dx; }

double Grid::dy() const { return _domain.dy; }

const Domain &Grid::domain() const { return _domain; }

bool Grid::getUseTemp() {return _useTemp;}

void Grid::setUseTemp(bool useTemp) {_useTemp = useTemp;}

const std::vector<Cell *> &Grid::fluid_cells() const { return _fluid_cells; }

const std::vector<Cell *> &Grid::fixed_wall_cells() const { return _fixed_wall_cells; }

const std::vector<Cell *> &Grid::fixed_wall_heat_cells() const { return _fixed_wall_heat_cells; }

const std::vector<Cell *> &Grid::fixed_wall_cold_cells() const { return _fixed_wall_cold_cells; }

const std::vector<Cell *> &Grid::moving_wall_cells() const { return _moving_wall_cells; }

const std::vector<Cell *> &Grid::inflow_wall_cells() const { return _inflow_wall_cells; }

const std::vector<Cell *> &Grid::outflow_wall_cells() const { return _outflow_wall_cells; }

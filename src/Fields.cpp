
#include <algorithm>
#include <iostream>

#include "Communication.hpp"
#include "Fields.hpp"
#include "math.h"

Fields::Fields(Grid &grid, double nu, double dt, double tau, double alpha, double beta, double UI, double VI,
               double PI, double TI, double GX, double GY)
    : _nu(nu), _dt(dt), _tau(tau), _alpha(alpha), _beta(beta), _gx(GX), _gy(GY) {
    _U = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);
    _V = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);
    _P = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);
    _T = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);

    _F = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);
    _G = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);
    _RS = Matrix<double>(grid.size_x() + 2, grid.size_y() + 2, 0.0);

    // TODO: think if the initialization is correct or if more needs to be done like this as in the other branch
    for (const auto &elem : grid.fluid_cells()) {
        int i = elem->i();
        int j = elem->j();

        _U(i, j) = UI;
        _V(i, j) = VI;
        _P(i, j) = PI;
        _T(i, j) = TI;
    }
}

void Fields::calculate_fluxes(Grid &grid) {

    for (const auto &elem : grid.fluid_cells()) {
        int i = elem->i();
        int j = elem->j();
        if (i != 0 && i != grid.size_x() +  1 && j != grid.size_x() + 1) 
        {
            _F(i, j) = _U(i, j) + _dt * ((_nu * Discretization::laplacian(_U, i, j)) -
                                        Discretization::convection_u(_U, _V, i, j) + _gx);
            _G(i, j) = _V(i, j) + _dt * ((_nu * Discretization::laplacian(_V, i, j)) -
                                        Discretization::convection_v(_U, _V, i, j) + _gy);

            if (grid.getUseTemp()) {
                // if the temperature calculation is used, we need to subtract the gx/gy terms again
                _F(i,j) -= _gx * _dt;
                _G(i,j) -= _gy * _dt;

                // add the boussinesq approximation to F and G
                _F(i,j) -= _beta * (_dt / 2)*(_T(i,j) + _T(i+1,j))*_gx;
                _G(i,j) -= _beta * (_dt / 2)*(_T(i,j) + _T(i,j+1))*_gy;
            }
        }
    }
}

void Fields::calculate_rs(Grid &grid) {
    for (const auto &elem : grid.fluid_cells()) {
        int i = elem->i();
        int j = elem->j();

        if (i != 0 && j != 0 && i != grid.size_x() + 1 && j != grid.size_y() + 1) { // exclude the buffer cells
        // calculate right hand side of PPE with F and G
            rs(i,j) = 1/_dt * ( (_F(i,j) - _F(i-1,j))/grid.dx() + (_G(i,j) - _G(i,j-1))/grid.dy() );
        }
    }
}

void Fields::calculate_velocities(Grid &grid) {
    for (const auto &elem : grid.fluid_cells()) {
        int i = elem->i();
        int j = elem->j();

        if (i != 0 && j != 0 && i != grid.size_x() + 1 && j != grid.size_y() + 1){ // exclude the buffer cells
        // update u
            u(i,j) = _F(i,j) - ((_dt)/(grid.dx())) * ( _P(i+1,j) - _P(i,j) );
        

        // update v
            v(i,j) = _G(i,j) - ((_dt)/(grid.dy())) * ( _P(i,j+1) - _P(i,j) );
        }
    }
}

void Fields::calculate_temperatures(Grid &grid) {
    // create new variable _Ttmp to store intermediate results, as e.g. in the convective term, we use T(i-1,j) to
    // calculate T(i,j) and we do not like to use the updated value already
    Matrix<double> _Ttmp{grid.size_x()+2, grid.size_y()+2};
    for (const auto &elem : grid.fluid_cells()) {
        int i = elem->i();
        int j = elem->j();
        if (i != 0 && j != 0 && i != grid.size_x() + 1 && j != grid.size_y() + 1) // exclude the buffer cells
        {   
            _Ttmp(i, j) = _T(i, j) + _dt * (_alpha * Discretization::laplacian(_T, i, j) -
                                        Discretization::convection_t(_U, _V, _T, i, j));
        }
    }

    // call copy constructor
    _T = _Ttmp;
}

double Fields::calculate_dt(Grid &grid) {
    
    double umax = 0.0;
    double vmax = 0.0;

    for (const auto &elem : grid.fluid_cells()) {
        int i = elem->i();
        int j = elem->j();

        if ( std::fabs(_U(i,j)) > umax ){ umax = std::fabs(_U(i,j)); }
        if ( std::fabs(_V(i,j)) > vmax ){ vmax = std::fabs(_V(i,j)); }
    }

    double arg1 = ( 1/(2*_nu) ) * (1 / ( (1/(grid.dx() * grid.dx())) + (1/(grid.dy() * grid.dy())) ));
    double arg2 = (grid.dx())/(umax);
    double arg3 = (grid.dy())/(vmax);

    _dt = std::min(arg1, std::min(arg2, arg3));
    if (grid.getUseTemp()) {
        double arg4 = (1/(2*_alpha)) * (1 / ( (1/(grid.dx() * grid.dx())) + (1/(grid.dy() * grid.dy())) ));
        _dt = std::min(_dt, arg4);
    }

    _dt *= _tau;

    if (_dt < 0.0005) {
        std::cout << "dt may get really small : " << _dt << std::endl;
        std::cout << "vamx: " << vmax << ", umax: " << umax << std::endl;
    }
    return _dt;
}

double &Fields::p(int i, int j) { return _P(i, j); }
double &Fields::u(int i, int j) { return _U(i, j); }
double &Fields::v(int i, int j) { return _V(i, j); }
double &Fields::t(int i, int j) { return _T(i, j); }
double &Fields::f(int i, int j) { return _F(i, j); }
double &Fields::g(int i, int j) { return _G(i, j); }

double &Fields::rs(int i, int j) { return _RS(i, j); }

Matrix<double> &Fields::u_matrix() { return _U; }
Matrix<double> &Fields::v_matrix() { return _V; }
Matrix<double> &Fields::t_matrix() { return _T; }
Matrix<double> &Fields::p_matrix() { return _P; }
Matrix<double> &Fields::f_matrix() { return _F; }
Matrix<double> &Fields::g_matrix() { return _G; }

double Fields::dt() const { return _dt; }

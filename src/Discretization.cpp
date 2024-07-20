#include "Discretization.hpp"

#include <cmath>

double Discretization::_dx = 0.0;
double Discretization::_dy = 0.0;
double Discretization::_gamma = 0.0;

Discretization::Discretization(double dx, double dy, double gamma) {
    _dx = dx;
    _dy = dy;
    _gamma = gamma;
}

double Discretization::convection_u(const Matrix<double> &U, const Matrix<double> &V, int i, int j) {

    double du2dx;
    double duvdy;

    // Given by the lecture notes
    du2dx = (0.25/_dx) * ( ( (U(i,j) + U(i+1,j)) * (U(i,j) + U(i+1,j)) ) - (U(i-1,j) + U(i,j)) * (U(i-1,j) + U(i,j))) ;
    du2dx += (_gamma * 0.25 / _dx) * ( std::abs(U(i,j) + U(i+1,j)) * ( U(i,j) - U(i+1,j) ) - std::abs(U(i-1,j) + U(i,j) ) * ( U(i-1,j) - U(i,j) ) );

    // Replace first Us' with Vs' in above calculation given by the lecture notes
    duvdy = (0.25/_dy) * ( (V(i, j) + V(i + 1, j)) * (U(i, j) + U(i, j + 1)) - (V(i, j - 1) + V(i + 1, j - 1)) * (U(i, j - 1) + U(i, j)) );
    duvdy += (_gamma * 0.25 / _dy) * (std::abs(V(i, j) + V(i + 1, j)) * (U(i, j) - U(i, j + 1)) - std::abs(V(i, j - 1) + V(i + 1, j - 1)) * (U(i, j - 1) - U(i, j)));

    double linear_combination = du2dx + duvdy;

    return linear_combination;

}

double Discretization::convection_v(const Matrix<double> &U, const Matrix<double> &V, int i, int j) {

    double duvdx;
    double dv2dy;

    dv2dy = (V(i, j) + V(i, j + 1)) * (V(i, j) + V(i, j + 1)) - (V(i, j - 1) + V(i, j)) * (V(i, j - 1) + V(i, j));
    dv2dy += _gamma * (std::abs(V(i, j) + V(i, j + 1)) * (V(i, j) - V(i, j + 1)) - std::abs(V(i, j - 1) + V(i, j)) * (V(i, j - 1) - V(i, j)));
    dv2dy *= (0.25 / _dy);

    duvdx = (V(i, j) + V(i + 1, j)) * (U(i, j) + U(i, j + 1)) -  (V(i - 1, j) + V(i, j)) * (U(i - 1, j) + U(i - 1, j + 1));
    duvdx += _gamma * (std::abs(U(i, j) + U(i, j + 1)) * (V(i, j) - V(i + 1, j)) - std::abs(U(i - 1, j) + U(i - 1, j + 1)) * (V(i - 1, j) - V(i, j)));
    duvdx *= (0.25 / _dx);

    double linear_combination = duvdx + dv2dy;

    return linear_combination;

}

double Discretization::convection_t(const Matrix<double> &U, const Matrix<double> &V, const Matrix<double> &T, int i, int j) {
    double duTdx = U(i, j) * (T(i, j) + T(i + 1, j)) - U(i - 1, j) * (T(i - 1, j) + T(i, j));
    duTdx += _gamma * (std::abs(U(i, j)) * (T(i, j) - T(i + 1, j)) - std::abs(U(i - 1, j)) * (T(i - 1, j) - T(i, j)));
    duTdx *= (0.5 / _dx);

    double dvTdy = V(i, j) * (T(i, j) + T(i, j + 1)) - V(i, j - 1) * (T(i, j - 1) + T(i, j));
    dvTdy += _gamma * (std::abs(V(i, j)) * (T(i, j) - T(i, j + 1)) - std::abs(V(i, j - 1)) * (T(i, j - 1) - T(i, j)));
    dvTdy *= (0.5 / _dy);

    double linear_combination{duTdx + dvTdy};
    return linear_combination;
}

double Discretization::laplacian(const Matrix<double> &A, int i, int j) {
    double laplacian_A = (A(i + 1, j) - 2.0 * A(i, j) + A(i - 1, j)) / (_dx * _dx) + (A(i, j + 1) - 2.0 * A(i, j) + A(i, j - 1)) / (_dy * _dy);
    return laplacian_A;
}

double Discretization::sor_helper(const Matrix<double> &P, int i, int j) {
    double laplacian_P_without_i_j = (P(i + 1, j) + P(i - 1, j)) / (_dx * _dx) + (P(i, j + 1) + P(i, j - 1)) / (_dy * _dy);
    return laplacian_P_without_i_j;
}

double Discretization::interpolate(const Matrix<double> &A, int i, int j, int i_offset, int j_offset) {
    double value = (A(i,j) + A(i+i_offset, j+j_offset))/2;
    return value; 
}

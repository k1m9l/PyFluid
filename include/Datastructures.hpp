#pragma once

#include <vector>

/**
 * @brief General 2D data structure around std::vector, in column
 * major format.
 *
 */
template <typename T> class Matrix {

  public:
    Matrix<T>() = default;

    /**
     * @brief Constructor with initial value
     *
     * @param[in] number of elements in x direction
     * @param[in] number of elements in y direction
     * @param[in] initial value for the elements
     *
     */
    Matrix<T>(int num_cols, int num_rows, double init_val) : _num_cols(num_cols), _num_rows(num_rows) {
        _container.resize(num_cols * num_rows);
        std::fill(_container.begin(), _container.end(), init_val);
    }

    /**
     * @brief Constructor without an initial value.
     *
     * @param[in] number of elements in x direction
     * @param[in] number of elements in y direction
     *
     */
    Matrix<T>(int num_cols, int num_rows) : _num_cols(num_cols), _num_rows(num_rows) {
        _container.resize(num_cols * num_rows);
    }

    /**
     * @brief Element access and modify using index
     *
     * @param[in] x index
     * @param[in] y index
     * @param[out] reference to the value
     */
    T &operator()(int i, int j) { return _container.at(_num_cols * j + i); }

    /**
     * @brief Element access using index
     *
     * @param[in] x index
     * @param[in] y index
     * @param[out] value of the element
     */
    T operator()(int i, int j) const { return _container.at(_num_cols * j + i); }

    /**
     * @brief Pointer representation of underlying data
     *
     * @param[out] pointer to the beginning of the vector
     */
    const T *data() const { return _container.data(); }

    /**
     * @brief Access of the size of the structure
     *
     * @param[out] size of the data structure
     */
    int size() const { return _container.size(); }

    /// get the given row of the matrix
    std::vector<double> get_row(int row) {
        std::vector<T> row_data(_num_cols, -1);
        for (int i = 0; i < _num_cols; ++i) {
            row_data.at(i) = _container.at(i + _num_cols * row);
        }
        return row_data;
    }

    /// get the given column of the matrix
    std::vector<double> get_col(int col) {
        std::vector<T> col_data(_num_rows, -1);
        for (int i = 0; i < _num_rows; ++i) {
            col_data.at(i) = _container.at(col + i * _num_cols);
        }
        return col_data;
    }

    /// set the given column of matrix to given vector
    void set_col(const std::vector<double> &vec, int col) {
        for (int i = 0; i < _num_rows; ++i) {
            _container.at(col + i * _num_cols) = vec.at(i);
        }
    }

    /// set the given row of matrix to given vector
    void set_row(const std::vector<double> &vec, int row) {
        for (int i = 0; i < _num_cols; ++i) {
            _container.at(i + row * _num_cols) = vec.at(i);
        }
    }

    /// get the number of elements in x direction
    int num_cols() const { return _num_cols; }

    /// get the number of elements in y direction
    int num_rows() const { return _num_rows; }

  private:
    /// Number of elements in x direction
    int _num_cols;
    /// Number of elements in y direction
    int _num_rows;

    /// Data container
    std::vector<T> _container;
};

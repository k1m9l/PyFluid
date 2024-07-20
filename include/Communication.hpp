#include "Fields.hpp"
#include <mpi.h>
#include <array>
class Communication {

  public:


    Communication() = default;

    static void init_parallel(int *argn, char **args, int &rank, int &size);

    static void finalize();

    static void communicate(Matrix<double> &data, const Domain &domain, int rank);

    static double reduce_min(double local_min);

    static double reduce_sum(double partial_sum);

    //static void communicate();

    //static double reduce_min(double local_min);

    //static double reduce_sum(double partial_sum);

   //static int _rank;
    //static int _size;
    //static int _iProc;
    //static int _jProc;

    //static int _my_coords[2];
    //static int _neighbours_ranks[4];
    //static MPI_Comm _comm;

};
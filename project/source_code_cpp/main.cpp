#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include "Case.hpp"
#include <chrono>

int main(int argn, char **args) {

    int rank, size;

    auto start = std::chrono::steady_clock::now();

    std::ofstream logging;
    std::stringstream ss;
    ss << "Simulation_log.txt";
    std::string loggingFilename;
    ss >> loggingFilename;
    logging.open(loggingFilename);

    Communication::init_parallel(&argn, args, rank, size);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argn > 1) {
        std::string file_name{args[1]};

        Case problem(file_name, argn, args, rank, size);

        problem.simulate(logging);
        Communication::finalize();

    } else {
        std::cout << "Error: No input file is provided to fluidchen." << std::endl;
        std::cout << "Example usage: /path/to/fluidchen /path/to/input_data.dat" << std::endl;
    }

    if(rank==0){
        //std::cout << "\nFinished.\n";
        //logging << "\nFinished.\n";
        auto end = std::chrono::steady_clock::now();
        std::cout << "Running Time:" << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s\n\n";
        logging << "Running Time:" << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s\n\n";
        logging.close();
    }
}

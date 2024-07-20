import json
import sys
import os
import time
from Case import Case
from mpi4py import MPI
from pyinstrument import Profiler


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()

    if len(sys.argv) > 1:
        file_name = sys.argv[1]

        problem = Case(file_name, rank, size)

        problem.simulate()
        MPI.Finalize()

    else:
        print("Error: No input file is provided to fluidchen.")
        print("Example usage: /path/to/fluidchen /path/to/input_data.dat")

    end = time.time()
    print(f"\nRunning Time: {int(end - start)}s\n\n")

if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    main()
    profiler.stop()

    output_name = sys.argv[2]


    # Get the HTML output
    html_output = profiler.output_html()

    code = sys.argv[3]

    output_string = output_name + code + ".html"

    # Save the HTML output to a file
    with open(output_string, 'w', encoding='utf-8') as f:
        f.write(html_output)

        

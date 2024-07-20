import mpi4py.MPI as MPI
import numpy as np

class Communication:

    @staticmethod
    def init_parallel(argn, args):
        MPI.Init(args)
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        return rank, size
    

    @staticmethod
    def finalize():
        MPI.Finalize

    @staticmethod
    def communicate(data, domain, rank):
        comm = MPI.COMM_WORLD

        # Communicate to Left
        if domain.neighbours[0] != -1:
            send = np.ascontiguousarray(data[1, :])
            recv = np.ascontiguousarray(np.zeros(domain.size_y + 2))
            comm.Sendrecv(send, dest=domain.neighbours[0], sendtag=1,
                          recvbuf=recv, source=domain.neighbours[0], recvtag=2)
            data[0, :] = recv

        # Communicate to Right
        if domain.neighbours[1] != -1:
            send = np.ascontiguousarray(data[domain.size_x, :])
            recv = np.ascontiguousarray(np.zeros(domain.size_y + 2))
            comm.Sendrecv(send, dest=domain.neighbours[1], sendtag=2,
                          recvbuf=recv, source=domain.neighbours[1], recvtag=1)
            data[domain.size_x + 1, :] = recv

        # Communicate to Top
        if domain.neighbours[2] != -1:
            send = np.ascontiguousarray(data[:, domain.size_y])
            recv = np.ascontiguousarray(np.zeros(domain.size_x + 2))
            comm.Sendrecv(send, dest=domain.neighbours[2], sendtag=3,
                          recvbuf=recv, source=domain.neighbours[2], recvtag=4)
            data[:, domain.size_y + 1] = recv

        # Communicate to Bottom
        if domain.neighbours[3] != -1:
            send = np.ascontiguousarray(data[:, 1])
            recv = np.ascontiguousarray(np.zeros(domain.size_x + 2))
            comm.Sendrecv(send, dest=domain.neighbours[3], sendtag=4,
                          recvbuf=recv, source=domain.neighbours[3], recvtag=3)
            data[:, 0] = recv


    @staticmethod
    def reduce_min(local_min):
        global_min = MPI.COMM_WORLD.allreduce(local_min, op=MPI.MIN)
        return global_min

    @staticmethod
    def reduce_sum(partial_sum):
        total_sum = MPI.COMM_WORLD.allreduce(partial_sum, op=MPI.SUM)
        return total_sum



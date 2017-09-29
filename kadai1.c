#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int myid, numproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    printf("process [%d]\n", myid);
    if (myid == 0) {
        printf("# of process [%d]\n", numproc);
    }

    MPI_Finalize();
    return 0;
}

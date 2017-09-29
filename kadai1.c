#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int myid, numproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(myid == 0) {
        int n = argc[0];
        int p = numproc;
        srand((unsigned)time(NULL));
        int nums[n];
        for (int i=0; i<n; i++) nums[i] = rand();

    } else {
    }

    MPI_Finalize();
    return 0;
}

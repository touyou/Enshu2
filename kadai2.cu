#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#define TIMEMAX 30
#define XSIZE 50
#define YSIZE 50

/*__global__ void add(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < n) {
        y[idx] = y[idx] + x[idx];
        idx = blockDim.x * gridDim.x + idx;
    }
}*/

__global__ void simmGpu(float ***u, float r) {
}

void simmCpu(float (*u)[2][XSIZE][YSIZE], float r) {
    int t, i, j;
    omp_set_num_threads(8);

    for (t=0; t<TIMEMAX; t++) {
        #pragma omp parallel for
        for (i=1; i<XSIZE-1; i++) {
            #pragma omp parallel for
            for (j=1; j<YSIZE-1; j++) {
                (*u)[(t+1)%2][i][j] = (1.0 - 4.0*r) * (*u)[t%2][i][j] + r * ((*u)[t%2][i+1][j] + (*u)[t%2][i-1][j] + (*u)[t%2][i][j+1] + (*u)[t%2][i][j-1]);
            }
        }
    }
}

int main() {
    struct timeval t0, t1;
    float ***devA;
    float u[2][XSIZE][YSIZE];
    int nb = 2 * XSIZE * YSIZE * sizeof(float), i, j;
    for (i=1; i<XSIZE-1; i++) {
        for (j=1; j<YSIZE; j++) {
            u[0][i][j] = 1;
        }
    }

    gettimeofday(&t0, NULL);
    simmCpu(&u, 0.12);
    /*cudaMalloc((void****)&devA, nb);
    cudaMemcpy(devA, a, nb, cudaMemcpyHostToDevice);
    add<<<2, 3>>>(devA, devA, 8);
    cudaMemcpy(a, devA, nb, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();*/
    gettimeofday(&t1, NULL);
    printf("Elapsed time = %lf\n", (double)(t1.tv_sec-t0.tv_sec)+(double)(t1.tv_usec-t0.tv_usec)*1.0e-6);
    for (i=0; i<XSIZE; i++) {
        for (j=0; j<YSIZE; j++) {
            printf("%.2lf ", u[0][i][j]);
        }
        puts("");
    }
}

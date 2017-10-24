#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#define TIMEMAX 100
#define XSIZE 50
#define YSIZE 50


/*__global__ void add(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < n) {
        y[idx] = y[idx] + x[idx];
        idx = blockDim.x * gridDim.x + idx;
    }
}*/

// blockIdx, blockDim, threadIdx, gridDim
__global__ void simmGpu(float *u, float r) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = id % XSIZE;
    int idy = id / XSIZE;
    int t;

    if (idx == 0 || idy == 0 || idx == XSIZE-1 || idy == YSIZE-1) return;

    for (t=0; t<TIMEMAX; t++) {
        float u1 = u[(t%2)*XSIZE*YSIZE+idx*YSIZE+idy];
        float u2 = u[(t%2)*XSIZE*YSIZE+(idx+1)*YSIZE+idy];
        float u3 = u[(t%2)*XSIZE*YSIZE+(idx-1)*YSIZE+idy];
        float u4 = u[(t%2)*XSIZE*YSIZE+idx*YSIZE+idy+1];
        float u5 = u[(t%2)*XSIZE*YSIZE+idx*YSIZE+idy-1];

        __syncthreads();
        u[((t+1)%2)*XSIZE*YSIZE+idx*YSIZE+idy] = (1.0 - 4.0*r) * u1 + r * (u2 + u3 + u4 + u5);
        __syncthreads();
    }
}

void simmCpu(float u[2][XSIZE][YSIZE], float r) {
    int t, i, j;
    omp_set_num_threads(8);

    for (t=0; t<TIMEMAX; t++) {
        #pragma omp parallel for
        for (i=1; i<XSIZE-1; i++) {
            #pragma omp parallel for
            for (j=1; j<YSIZE-1; j++) {
                u[(t+1)%2][i][j] = (1.0 - 4.0*r) * u[t%2][i][j] + r * (u[t%2][i+1][j] + u[t%2][i-1][j] + u[t%2][i][j+1] + u[t%2][i][j-1]);
            }
        }
    }
}

int divRoundUp(int value, int radix) {
    return (value + radix - 1) / radix;
}

int main() {
    struct timeval t0, t1;
    float *devA;
    float u[2][XSIZE][YSIZE];
    int nb = 2 * XSIZE * YSIZE * sizeof(float), i, j;
    memset(u, 0, nb);
    for (i=1; i<XSIZE-1; i++) {
        for (j=1; j<YSIZE-1; j++) {
            u[0][i][j] = 1;
        }
    }

    cudaMalloc((void**)&devA, nb);
    cudaMemcpy(devA, u, nb, cudaMemcpyHostToDevice);
    gettimeofday(&t0, NULL);
    simmGpu<<<XSIZE, YSIZE>>>(devA, 0.12);
    gettimeofday(&t1, NULL);
    cudaMemcpy(u, devA, nb, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    printf("GPU: Elapsed time = %lf\n", (double)(t1.tv_sec-t0.tv_sec)+(double)(t1.tv_usec-t0.tv_usec)*1.0e-6);

    for (i=0; i<XSIZE; i++) {
        for (j=0; j<YSIZE; j++) {
            if (u[0][i][j] > 0.34) printf("#");
            else if (u[0][i][j] > -0.34) printf("*");
            else printf(".");
        }
        puts("");
    }

    memset(u, 0, nb);
    for (i=1; i<XSIZE-1; i++) {
        for (j=1; j<YSIZE-1; j++) {
            u[0][i][j] = 1;
        }
    }

    gettimeofday(&t0, NULL);
    simmCpu(u, 0.12);
    gettimeofday(&t1, NULL);
    printf("openMP: Elapsed time = %lf\n", (double)(t1.tv_sec-t0.tv_sec)+(double)(t1.tv_usec-t0.tv_usec)*1.0e-6);

    for (i=0; i<XSIZE; i++) {
        for (j=0; j<YSIZE; j++) {
            if (u[0][i][j] > 0.34) printf("#");
            else if (u[0][i][j] > -0.34) printf("*");
            else printf(".");
        }
        puts("");
    }

}

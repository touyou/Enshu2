#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int int_sort(const void *a, const void *b) {
    if (*(int *)a < *(int *)b) return -1;
    else if (*(int *)a < *(int *)b) return 0;
    return 1;
}

int int_sort_desc(const void *a, const void *b) {
    if (*(int *)a > *(int *)b) return -1;
    else if (*(int *)a == *(int *)b) return 0;
    return 1;
}

// 双単調マージに基づくブロック奇遇ソート法
// http://www.dbl.k.hosei.ac.jp/~miurat/readings/Oct1805.pdf
// 1ステップは奇数番目なら奇数番目の、偶数番目なら偶数番目の右隣と双単調マージ交換を行う
// O(mp^2)ぐらい
// 双単調マージは昇順と降順を交互になるようにソートして各要素を比較して交換
// ステップの交換が終わったらまたソート
// ソートはクイックソートを使う

int main(int argc, char **argv) {
    int myid, numproc;
    int m = 10;
    int i, j;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int p = numproc;
    int nums[m];

    if(myid == 0) {
        int p = numproc;
        int n = m * p;
        srand((unsigned)time(NULL));
        int init_nums[n];
        for (i=0; i<n; i++) {
            nums[i] = rand();
            if (i < m) nums[i] = init_nums[i];
        }
        for (i=1; i<p; i++) {
            MPI_Send(nums + i * m, m, MPI_INT, i, 1000, MPI_COMM_WORLD);
        }
    } else {
        int nums[m];
        MPI_Recv(nums, m, MPI_INT, 0, 1000, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    int step;
    for (step=0; step<p; step++) {
        if (step % 2 == myid % 2 && myid != p - 1) {
            // 一致なら右隣と交換
            // この時は昇順
            qsort((void *)nums, m, sizeof(int), int_sort);
            for (i=0; i<m; i++) {
                int upper, lower = nums[i];
                MPI_Send(&lower, 1, MPI_INT, myid+1, i+myid, MPI_COMM_WORLD);
                MPI_Recv(&upper, 1, MPI_INT, myid+1, i+myid, MPI_COMM_WORLD, &status);
                if (upper < lower) nums[i] = upper;
            }
        } else if (step % 2 != myid %2 && myid != 0) {
            // 不一致なら左隣と交換
            // この時は降順
            qsort((void *)nums, m, sizeof(int), int_sort_desc);
            for (i=0; i<m; i++) {
                int lower, upper = nums[i];
                MPI_Send(&upper, 1, MPI_INT, myid-1, i+myid-1, MPI_COMM_WORLD);
                MPI_Recv(&lower, 1, MPI_INT, myid-1, i+myid-1, MPI_COMM_WORLD, &status);
                if (upper < lower) nums[i] = lower;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    double elapsed = t2 - t1; // かかった秒数
    // ソート終了。結果をすべて返していく
    if (myid == 0) {
        // 全部から受け取って、ソート結果と時間を表示
        printf("%.5fsec\n", elapsed);
        printf("process 0 ---\n");
        for (i=0; i<m; i++) {
            if (i != 0) printf(" ");
            printf("%d", nums[i]);
        }
        puts("");
        for (i=1; i<p; i++) {
            int recv[m];
            MPI_Recv(recv, m, MPI_INT, i, 100+i, MPI_COMM_WORLD, &status);
            printf("process %d---\n", i);
            for (j=0; j<m; j++) {
                if (j != 0) printf(" ");
                printf("%d", recv[j]);
            }
            puts("");
        }
    } else {
        // ソートしたものを0に渡す
        MPI_Send(nums, m, MPI_INT, 0, 100+myid, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

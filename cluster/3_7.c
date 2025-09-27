#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define PING_PONG_LIMIT 100000   // número de repeticiones
#define MENSAJE_SIZE 1               // tamaño del mensaje

int main(int argc, char* argv[]) {
    int rank, size;
    int mensaje = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int compa = (rank + 1) % 2;

    clock_t inicio, fin;
    double tiempo = 0.0;

    if (rank == 0) {
        inicio = clock();
        for (int i = 0; i < PING_PONG_LIMIT; i++) {
            MPI_Send(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD);
            MPI_Recv(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        fin = clock();
        tiempo = ((double)(fin - inicio)) / CLOCKS_PER_SEC;
        printf("Tiempo con clock(): %f segundos\n", tiempo);
    } else {
        for (int i = 0; i < PING_PONG_LIMIT; i++) {
            MPI_Recv(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_WTIME
    double inicio_wtime, fin_wtine, tiempo_wtime = 0.0;

    if (rank == 0) {
        inicio_wtime = MPI_Wtime();
        for (int i = 0; i < PING_PONG_LIMIT; i++) {
            MPI_Send(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD);
            MPI_Recv(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        fin_wtine = MPI_Wtime();
        tiempo_wtime = fin_wtine - inicio_wtime;
        printf("Tiempo con MPI_Wtime(): %f segundos\n", tiempo_wtime);
        printf("Latencia promedio por ping-pong = %e segundos\n", tiempo_wtime / PING_PONG_LIMIT);
    } else {
        for (int i = 0; i < PING_PONG_LIMIT; i++) {
            MPI_Recv(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&mensaje, MENSAJE_SIZE, MPI_INT, compa, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}

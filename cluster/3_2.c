#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>   // Para gethostname


int main(int argc, char* argv[]) {
    int rank, comm_sz;
    long long int total, local;
    long long int number_in_circle_local = 0, number_in_circle_total = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Proceso %d de %d corriendo en %s\n", rank, comm_sz, hostname);

    if (rank == 0) {
        printf("Ingrese el numero total de lanzamientos: ");
        scanf("%lld", &total);
    }

    MPI_Bcast(&total, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    local = total / comm_sz;
    srand(time(NULL) + rank);

    for (long long int toss = 0; toss < local; toss++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1,1]
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double dist2 = x*x + y*y;
        if (dist2 <= 1.0) number_in_circle_local++;
    }

    MPI_Reduce(&number_in_circle_local, &number_in_circle_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double pi = 4.0 * ((double) number_in_circle_total / (double) total);
        printf("Estimacion de pi = %.10f\n", pi);
    }

    MPI_Finalize();
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size;
    long long int total_tosses, local_tosses;
    long long int number_in_circle_local = 0, number_in_circle_total = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Ingrese el numero total de lanzamientos: ");
        scanf("%lld", &total_tosses);
    }

    MPI_Bcast(&total_tosses, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    local_tosses = total_tosses / size;
    srand(time(NULL) + rank);

    for (long long int toss = 0; toss < local_tosses; toss++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0; // [-1,1]
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double dist2 = x*x + y*y;
        if (dist2 <= 1.0) number_in_circle_local++;
    }

    MPI_Reduce(&number_in_circle_local, &number_in_circle_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double pi_estimate = 4.0 * ((double) number_in_circle_total / (double) total_tosses);
        printf("Estimacion de pi = %.10f\n", pi_estimate);
    }

    MPI_Finalize();
    return 0;
}


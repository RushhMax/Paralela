
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[256];
    gethostname(hostname, 256);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hola desde rank %d de %d en %s\n", rank, size, hostname);

    MPI_Finalize();
    return 0;
}


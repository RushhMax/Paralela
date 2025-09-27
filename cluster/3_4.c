#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int rank, comm_sz;
    int valor_local, suma_global;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    srand(time(NULL));
    valor_local = rand() % 20 + 1;

    int salto = 1;
    suma_global = valor_local;

    while (salto < comm_sz) {
        int compa = rank ^ salto; 
        int recibe;

        MPI_Sendrecv(&suma_global, 1, MPI_INT, compa, 0,
                     &recibe, 1, MPI_INT, compa, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        suma_global += recibe;
        salto <<= 1;  
    }

    printf("Proceso %d: suma global = %d\n", rank, suma_global);

    MPI_Finalize();
    return 0;
}

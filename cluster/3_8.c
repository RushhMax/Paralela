#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

int compara(const void *a, const void *b) {
    int A = *(int*)a;
    int B = *(int*)b;
    return (A - B);
}

int* merge(int *a, int comm_sz_a, int *b, int comm_sz_b) {
    int *result = malloc((comm_sz_a + comm_sz_b) * sizeof(int));
    int i = 0, j = 0, k = 0;

    while (i < comm_sz_a && j < comm_sz_b) {
        if (a[i] <= b[j]) {
            result[k++] = a[i++];
        } else {
            result[k++] = b[j++];
        }
    }
    while (i < comm_sz_a) result[k++] = a[i++];
    while (j < comm_sz_b) result[k++] = b[j++];
    return result;
}

int main(int argc, char* argv[]) {
    int rank, comm_sz;
    int n; 
    int *localD;
    int localN;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Proceso %d de %d corriendo en %s\n", rank, comm_sz, hostname);


    if (rank == 0) {
        if (argc != 2) {
            printf("Uso: %s <número de elementos>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = atoi(argv[1]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    localN = n / comm_sz;
    localD = malloc(localN * sizeof(int));

    srand(time(NULL) + rank);
    for (int i = 0; i < localN; i++) {
        localD[i] = rand() % 100; 
    }

    qsort(localD, localN, sizeof(int), compara);
    int *all_data = NULL;
    if (rank == 0) {
        all_data = malloc(n * sizeof(int));
    }
    MPI_Gather(localD, localN, MPI_INT,
               all_data, localN, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Listas locales (concatenadas):\n");
        for (int i = 0; i < n; i++) {
            printf("%d ", all_data[i]);
        }
        printf("\n\n");
        free(all_data);
    }

    int salto = 1;
    int *resultado = localD;
    int resultado_comm_sz = localN;

    while (salto < comm_sz) {
        if (rank % (2*salto) == 0) {
            int compa = rank + salto;
            if (compa < comm_sz) {
                int recibe_comm_sz;
                MPI_Recv(&recibe_comm_sz, 1, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int *recibe_data = malloc(recibe_comm_sz * sizeof(int));
                MPI_Recv(recibe_data, recibe_comm_sz, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int *nuevo = merge(resultado, resultado_comm_sz, recibe_data, recibe_comm_sz);
                free(resultado);
                free(recibe_data);
                resultado = nuevo;
                resultado_comm_sz += recibe_comm_sz;
            }
        } else {
            // Este proceso envía sus datos a su compañero y sale del bucle
            int compa = rank - salto;
            MPI_Send(&resultado_comm_sz, 1, MPI_INT, compa, 0, MPI_COMM_WORLD);
            MPI_Send(resultado, resultado_comm_sz, MPI_INT, compa, 0, MPI_COMM_WORLD);
            free(resultado);
            break;
        }
        salto *= 2;
    }

    if (rank == 0) {
        printf("Lista global ordenada:\n");
        for (int i = 0; i < n; i++) {
            printf("%d ", resultado[i]);
        }
        printf("\n");
        free(resultado);
    }

    MPI_Finalize();
    return 0;
}

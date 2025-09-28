#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int Find_bin(float value, float min_meas, int bin_count, float bin_maxes[]) {
    int bin;
    for (bin = 0; bin < bin_count; bin++) {
        if (value < bin_maxes[bin]) {
            return bin;
        }
    }
    // Si llega aquí, pertenece al último bin
    return bin_count - 1;
}

int main(int argc, char* argv[]) {
    int rank, comm_sz;
    int data_count, bin_count;
    float min_meas, max_meas, bin_width;
    float *data = NULL;
    float *local_data;
    float *bin_maxes;
    int *loc_bin_cts, *bin_counts;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (rank == 0) {
        // Leer parámetros
        printf("Número de datos: ");
        scanf("%d", &data_count);

        printf("Valor mínimo: ");
        scanf("%f", &min_meas);

        printf("Valor máximo: ");
        scanf("%f", &max_meas);

        printf("Número de bins: ");
        scanf("%d", &bin_count);

        // Reservar memoria
        data = malloc(data_count * sizeof(float));
        bin_maxes = malloc(bin_count * sizeof(float));
        bin_counts = malloc(bin_count * sizeof(int));

        // Leer datos
        printf("Ingrese los %d valores:\n", data_count);
        for (int i = 0; i < data_count; i++) {
            scanf("%f", &data[i]);
        }

        // Calcular ancho de bins y bin_maxes
        bin_width = (max_meas - min_meas) / bin_count;
        for (int b = 0; b < bin_count; b++) {
            bin_maxes[b] = min_meas + bin_width * (b + 1);
        }
    }

    // Broadcast de parámetros
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        bin_maxes = malloc(bin_count * sizeof(float));
    }

    MPI_Bcast(&min_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bin_maxes, bin_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Distribuir datos
    int local_n = data_count / comm_sz;
    local_data = malloc(local_n * sizeof(float));

    MPI_Scatter(data, local_n, MPI_FLOAT,
                local_data, local_n, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Inicializar histogramas locales
    loc_bin_cts = malloc(bin_count * sizeof(int));
    for (int b = 0; b < bin_count; b++)
        loc_bin_cts[b] = 0;

    // Calcular histograma local
    for (int i = 0; i < local_n; i++) {
        int bin = Find_bin(local_data[i], min_meas, bin_count, bin_maxes);
        loc_bin_cts[bin]++;
    }

    // Reducir a histograma global
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, loc_bin_cts, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(loc_bin_cts, NULL, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Proceso 0 imprime
    if (rank == 0) {
        printf("\nHistograma final:\n");
        for (int b = 0; b < bin_count; b++) {
            printf("Bin %d [%.2f - %.2f): %d\n",
                   b,
                   (b == 0 ? min_meas : bin_maxes[b - 1]),
                   bin_maxes[b],
                   loc_bin_cts[b]);
        }
    }

    // Liberar memoria
    free(local_data);
    free(loc_bin_cts);
    free(bin_maxes);
    if (rank == 0) {
        free(data);
        free(bin_counts);
    }

    MPI_Finalize();
    return 0;
}


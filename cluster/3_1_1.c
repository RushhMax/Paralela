//mpicc mpi_histograma.c -o mpi_histograma
//mpirun -np 3 -host nodo1,nodo2,nodo3 ./mpi_histograma


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int find_bin(float value, float min_meas, int bin_count, float bin_width) {
    int bin = (int)(value - min_meas) / bin_width;
    if(bin < 0) bin = 0;
    if (bin >= bin_count) bin = bin_count - 1; // valores en el borde
    return bin;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int data_count = 20; // número de datos (ejemplo)
    int bin_count = 5;   // número de bins
    float min_meas = 0.0, max_meas = 5.0;
    float bin_width = (max_meas - min_meas) / bin_count;

    float *data = NULL;        // datos completos (solo en rank 0)
    float *local_data = NULL;  // trozo de datos en cada proceso
    int *local_bin_counts = NULL;
    int *global_bin_counts = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = data_count / size; // elementos por proceso


    if (rank == 0) {
        // Datos de ejemplo (pueden leerse de archivo)
        float temp_data[20] = {1.3, 2.9, 0.4, 0.3, 1.3,
                               4.4, 1.7, 0.4, 3.2, 0.3,
                               4.9, 2.4, 3.1, 4.4, 3.9,
                               0.4, 4.2, 4.5, 4.9, 0.9};
        data = malloc(data_count * sizeof(float));
        for (int i = 0; i < data_count; i++)
            data[i] = temp_data[i];
    }

    // Reservar memoria local
    local_data = malloc(local_n * sizeof(float));
    local_bin_counts = calloc(bin_count, sizeof(int));

    // Distribuir datos entre procesos
    MPI_Scatter(data, local_n, MPI_FLOAT,
                local_data, local_n, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Cada proceso calcula su histograma local
    for (int i = 0; i < local_n; i++) {
        int bin = find_bin(local_data[i], min_meas, bin_count, bin_width);
        local_bin_counts[bin]++;
    }

    // Reunir histogramas locales en el proceso 0
    if (rank == 0) {
        global_bin_counts = calloc(bin_count, sizeof(int));
    }

    MPI_Reduce(local_bin_counts, global_bin_counts,
               bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Proceso 0 imprime el histograma final
    if (rank == 0) {
        printf("Histograma final:\n");
        for (int b = 0; b < bin_count; b++) {
            printf("Bin %d [%.2f - %.2f): %d\n",
                   b, min_meas + b*bin_width,
                   min_meas + (b+1)*bin_width,
                   global_bin_counts[b]);
        }
    }

    free(local_data);
    free(local_bin_counts);
    if (rank == 0) {
        free(data);
        free(global_bin_counts);
    }

    MPI_Finalize();
    return 0;
}

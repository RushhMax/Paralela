//mpicc mpi_histograma.c -o mpi_histograma
//mpirun -np 3 -host nodo1,nodo2,nodo3 ./mpi_histograma


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int find_bin(float value, float min_meas, int contad_bin, float bin_width) {
    int bin = (int)(value - min_meas) / bin_width;
    if(bin < 0) bin = 0;
    if (bin >= contad_bin) bin = contad_bin - 1;
    return bin;
}

int main(int argc, char* argv[]) {
    int rank, comm_sz;
    int contad_data = 20; 
    int contad_bin = 5;   
    float min_meas = 0.0, max_meas = 5.0;
    float bin_width = (max_meas - min_meas) / contad_bin;

    float *data = NULL;       
    float *local_data = NULL;
    int *local_contad_bins = NULL;
    int *global_contad_bins = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_comm_sz(MPI_COMM_WORLD, &comm_sz);

    int local_n = contad_data / comm_sz; 


    if (rank == 0) {
        float temp_data[20] = {1.3, 2.9, 0.4, 0.3, 1.3,
                               4.4, 1.7, 0.4, 3.2, 0.3,
                               4.9, 2.4, 3.1, 4.4, 3.9,
                               0.4, 4.2, 4.5, 4.9, 0.9};
        data = malloc(contad_data * comm_szof(float));
        for (int i = 0; i < contad_data; i++)
            data[i] = temp_data[i];
    }

    local_data = malloc(local_n * comm_szof(float));
    local_contad_bins = calloc(contad_bin, comm_szof(int));

    MPI_Scatter(data, local_n, MPI_FLOAT,
                local_data, local_n, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        int bin = find_bin(local_data[i], min_meas, contad_bin, bin_width);
        local_contad_bins[bin]++;
    }

    if (rank == 0) {
        global_contad_bins = calloc(contad_bin, comm_szof(int));
    }

    MPI_Reduce(local_contad_bins, global_contad_bins,
               contad_bin, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Histograma final:\n");
        for (int b = 0; b < contad_bin; b++) {
            printf("Bin %d [%.2f - %.2f): %d\n",
                   b, min_meas + b*bin_width,
                   min_meas + (b+1)*bin_width,
                   global_contad_bins[b]);
        }
    }

    free(local_data);
    free(local_contad_bins);
    if (rank == 0) {
        free(data);
        free(global_contad_bins);
    }

    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

static void compute_conteo_bloques_displs(int n, int p, int *counts, int *displs) {
    int base = n / p;
    int rem = n % p;
    int offset = 0;
    for (int i = 0; i < p; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
}

// devolver proceso dueño en distribución por bloques del índice global idx
static int owner_of_index_block(int idx, int n, int p) {
    int base = n / p;
    int rem = n % p;
    // indices 0..n-1, first rem processes have base+1 elements
    int boundary = (base + 1) * rem;
    if (idx < boundary) {
        return idx / (base + 1);
    } else {
        return rem + (idx - boundary) / base;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Uso: %s <n> [reps]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    long long n = atoll(argv[1]);
    int reps = (argc >= 3) ? atoi(argv[2]) : 10;
    if (n <= 0 || reps <= 0) {
        if (rank == 0) fprintf(stderr, "n y reps deben ser > 0\n");
        MPI_Finalize();
        return 1;
    }

    int *conteo_bloques = malloc(comm_sz * sizeof(int));
    int *block_displs = malloc(comm_sz * sizeof(int));
    compute_conteo_bloques_displs((int)n, comm_sz, conteo_bloques, block_displs);

    int local_block_n = conteo_bloques[rank];
    int local_start = block_displs[rank];

    // Vector local en distribución por bloques: contiene los índices globales (para verificar)
    int *local_block = malloc(local_block_n * sizeof(int));
    for (int i = 0; i < local_block_n; ++i) local_block[i] = local_start + i;

    // Preparar arrays auxiliares para Alltoallv (block -> cyclic)
    int *sendcounts = calloc(comm_sz, sizeof(int));
    int *sdispls = calloc(comm_sz, sizeof(int));
    int *recvcounts = calloc(comm_sz, sizeof(int));
    int *rdispls = calloc(comm_sz, sizeof(int));

    // calcular sendcounts: cuántos elementos locales van a cada destino (dest = idx % comm_sz)
    for (int i = 0; i < local_block_n; ++i) {
        int gidx = local_block[i];
        int dest = gidx % comm_sz; // en distribución cíclica el idx va a process (idx % P)
        sendcounts[dest]++;
    }
    // sdispls y tamaño del sendbuf
    sdispls[0] = 0;
    for (int p = 1; p < comm_sz; ++p) sdispls[p] = sdispls[p-1] + sendcounts[p-1];
    int total_send = sdispls[comm_sz-1] + sendcounts[comm_sz-1];
    int *sendbuf = malloc(total_send * sizeof(int));
    // llenar sendbuf por offsets
    int *pos = malloc(comm_sz * sizeof(int));
    for (int p = 0; p < comm_sz; ++p) pos[p] = sdispls[p];
    for (int i = 0; i < local_block_n; ++i) {
        int gidx = local_block[i];
        int dest = gidx % comm_sz;
        sendbuf[pos[dest]++] = gidx;
    }
    free(pos);

    // ahora exchange de counts para saber cuánto recibiremos
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
    rdispls[0] = 0;
    for (int p = 1; p < comm_sz; ++p) rdispls[p] = rdispls[p-1] + recvcounts[p-1];
    int total_recv = rdispls[comm_sz-1] + recvcounts[comm_sz-1];
    int *recvbuf = malloc(total_recv * sizeof(int));

    // --- medimos block -> cyclic (Solo la comunicación Alltoallv) ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int it = 0; it < reps; ++it) {
        // usar el mismo sendbuf; medir solo Alltoallv
        MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT,
                      recvbuf, recvcounts, rdispls, MPI_INT,
                      MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();
    double avg_block_to_cyclic = (t1 - t0) / reps;

    // Verificación: recvbuf debe contener global indices i such that i%comm_sz == rank
    int ok = 1;
    for (int i = 0; i < total_recv; ++i) {
        if (recvbuf[i] % comm_sz != rank) { ok = 0; break; }
    }
    int all_ok;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Block -> Cyclic: reps=%d, avg time = %g s  (all_ok=%d)\n",
               reps, avg_block_to_cyclic, all_ok);
    }

    // ------------------- ahora cyclic -> block (inversa) -------------------
    // Partimos de que recvbuf contiene los elementos que están en distribución cíclica
    // Cada process debe enviar cada elemento de recvbuf al owner del índice en la distribución por bloques.
    // calcular sendcounts2 (desde cyclic local elements hacia block owner)
    int *sendcounts2 = calloc(comm_sz, sizeof(int));
    int *sdispls2 = calloc(comm_sz, sizeof(int));
    int *recvcounts2 = calloc(comm_sz, sizeof(int));
    int *rdispls2 = calloc(comm_sz, sizeof(int));

    // preparar mapping owner_of_index_block
    for (int i = 0; i < total_recv; ++i) {
        int gidx = recvbuf[i];
        int dest = owner_of_index_block(gidx, (int)n, comm_sz);
        sendcounts2[dest]++;
    }
    sdispls2[0] = 0;
    for (int p = 1; p < comm_sz; ++p) sdispls2[p] = sdispls2[p-1] + sendcounts2[p-1];
    int total_send2 = sdispls2[comm_sz-1] + sendcounts2[comm_sz-1];
    int *sendbuf2 = malloc(total_send2 * sizeof(int));
    pos = malloc(comm_sz * sizeof(int));
    for (int p = 0; p < comm_sz; ++p) pos[p] = sdispls2[p];
    for (int i = 0; i < total_recv; ++i) {
        int gidx = recvbuf[i];
        int dest = owner_of_index_block(gidx, (int)n, comm_sz);
        sendbuf2[pos[dest]++] = gidx;
    }
    free(pos);

    // intercambiar counts
    MPI_Alltoall(sendcounts2, 1, MPI_INT, recvcounts2, 1, MPI_INT, MPI_COMM_WORLD);
    rdispls2[0] = 0;
    for (int p = 1; p < comm_sz; ++p) rdispls2[p] = rdispls2[p-1] + recvcounts2[p-1];
    int total_recv2 = rdispls2[comm_sz-1] + recvcounts2[comm_sz-1];
    int *recvbuf2 = malloc(total_recv2 * sizeof(int));

    // medir cyclic -> block
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    for (int it = 0; it < reps; ++it) {
        MPI_Alltoallv(sendbuf2, sendcounts2, sdispls2, MPI_INT,
                      recvbuf2, recvcounts2, rdispls2, MPI_INT,
                      MPI_COMM_WORLD);
    }
    double t3 = MPI_Wtime();
    double avg_cyclic_to_block = (t3 - t2) / reps;

    // Verificación: recvbuf2 debe contener exactly the block elements for this rank:
    // i in [block_displs[rank], block_displs[rank]+conteo_bloques[rank]-1]
    int ok2 = 1;
    if (total_recv2 != conteo_bloques[rank]) ok2 = 0;
    else {
        // mark presence
        for (int i = 0; i < total_recv2; ++i) {
            int g = recvbuf2[i];
            if (g < block_displs[rank] || g >= block_displs[rank] + conteo_bloques[rank]) { ok2 = 0; break; }
        }
    }
    int all_ok2;
    MPI_Allreduce(&ok2, &all_ok2, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Cyclic -> Block: reps=%d, avg time = %g s  (all_ok=%d)\n",
               reps, avg_cyclic_to_block, all_ok2);
    }

    // imprimir tiempos por proceso (opcional)
    double tvals[2] = {avg_block_to_cyclic, avg_cyclic_to_block};
    double tmin[2], tmax[2], tavg[2];
    for (int k = 0; k < 2; ++k) {
        MPI_Reduce(&tvals[k], &tmin[k], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tvals[k], &tmax[k], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tvals[k], &tavg[k], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) tavg[k] /= comm_sz;
    }
    if (rank == 0) {
        printf("Summary (seconds)  min  avg  max\n");
        printf("block->cyclic      %g  %g  %g\n", tmin[0], tavg[0], tmax[0]);
        printf("cyclic->block      %g  %g  %g\n", tmin[1], tavg[1], tmax[1]);
    }

    // cleanup
    free(conteo_bloques); free(block_displs);
    free(sendcounts); free(sdispls); free(recvcounts); free(rdispls);
    free(sendbuf); free(recvbuf);
    free(sendcounts2); free(sdispls2); free(recvcounts2); free(rdispls2);
    free(sendbuf2); free(recvbuf2);
    free(local_block);

    MPI_Finalize();
    return 0;
}

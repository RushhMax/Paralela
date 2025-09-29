#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char* argv[]){
  int rank, comm_sz;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("Proceso %d de %d corriendo en %s\n", rank, comm_sz, hostname);

  int n;

  double *M = NULL;
  double *V = NULL;
  if(rank == 0){
    printf("Orden de la matriz, que sea divisible por %d", comm_sz);
    scanf("%d", &n);
    M = (double*)malloc(n*n*sizeof(double));
    V = (double*)malloc(n*sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < n*n; i++) M[i] = rand()%5 + 1;
    for (int i = 0; i < n; i++) V[i] = rand()%5 + 1;

    printf("Matriz:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4.1f ", M[i*n + j]);
        }
        printf("\n");
    }

    printf("Vector:\n");
    for (int i = 0; i < n; i++) printf("%4.1f ", V[i]);
    printf("\n");
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int n_col = n/comm_sz;
  int tam_bloque = n*n_col;

  double *M_local = malloc(tam_bloque * sizeof(double));
  double *x_local = malloc(n_col * sizeof(double)); 
  double *resultado_local = calloc(n, sizeof(double)); // resultado parcial del proceso
  double *resultado_aux = malloc((n/comm_sz) * sizeof(double)); // vector resultado final

  MPI_Scatter(M, tam_bloque, MPI_DOUBLE, M_local, tam_bloque, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(V, n_col, MPI_DOUBLE, x_local, n_col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for(int i=0; i<n; i++){
    for(int j=0; j<n_col; j++){
      resultado_local[i] += M_local[i*n_col + j] * x_local[j];
    }
  }
 
  int resultados_count[comm_sz];
  for(int i=0; i<comm_sz; i++) resultados_count[i] = n/comm_sz;

  MPI_Reduce_scatter(resultado_local, resultado_aux, resultados_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("Rank %d tiene parte de y:\n", rank);
  for (int i = 0; i < n/comm_sz; i++) {
    printf("%6.2f ", resultado_aux[i]);
  }
  printf("\n");

  free(M_local); free(x_local); free(resultado_local); free(resultado_aux);
  if (rank == 0) { free(M); free(V); }

  MPI_Finalize();
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

int main(int argc, char* argv[]){
  int rank, comm_sz;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  int q = (int)sqrt(comm_sz);
  if (q*q != comm_sz) {
    if (rank == 0) printf("El número de procesos debe ser un cuadrado perfecto\n");
    MPI_Finalize();
    return 0;
  }


  int dims[2] = {q, q};
  int periods[2] = {0, 0};
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);
  int row = coords[0], col = coords[1];

  int n;
  double *M = NULL, *V = NULL;

  if (rank == 0) {
    printf("Orden de la matriz (divisible por %d): ", q);
    scanf("%d", &n);

    M = malloc(n*n*sizeof(double));
    V = malloc(n*sizeof(double));
    srand(time(NULL));

    for (int i = 0; i < n*n; i++) M[i] = rand()%5 + 1;
    for (int i = 0; i < n; i++) V[i] = rand()%5 + 1;

    printf("Matriz:\n");
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) printf("%4.0f ", M[i*n + j]);
      printf("\n");
    }
    printf("Vector:\n");
    for (int i = 0; i < n; i++) printf("%4.0f ", V[i]);
      printf("\n");
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int block_size = n/q;
  double *M_local = malloc(block_size*block_size*sizeof(double));
  double *V_local = malloc(block_size*sizeof(double));
  double *resultado_local = calloc(block_size, sizeof(double));
  
  if (rank == 0) {
    for (int pr = 0; pr < comm_sz; pr++) {
      int c[2];
      MPI_Cart_coords(grid_comm, pr, 2, c);
      int start_row = c[0] * block_size;
      int start_col = c[1] * block_size;

      double *temp = malloc(block_size*block_size*sizeof(double));
      for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
          temp[i*block_size+j] = M[(start_row+i)*n + (start_col+j)];

      if (pr == 0) {
        for (int i = 0; i < block_size*block_size; i++) M_local[i] = temp[i];
      } else {
        MPI_Send(temp, block_size*block_size, MPI_DOUBLE, pr, 0, MPI_COMM_WORLD);
      }
      free(temp);
    }
  } else {
    MPI_Recv(M_local, block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  if (row == col) {
    for (int i = 0; i < block_size; i++)
        V_local[i] = V[row*block_size + i];
    }
    MPI_Bcast(V_local, block_size, MPI_DOUBLE, row, grid_comm);
 
  for (int i = 0; i < block_size; i++) {
    for (int j = 0; j < block_size; j++) {
       resultado_local[i] += M_local[i*block_size + j] * V_local[j];
    }
  }
  double *y_final = NULL;
  if (rank == 0) y_final = malloc(n*sizeof(double));
  MPI_Gather(resultado_local, block_size, MPI_DOUBLE, y_final, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    printf("Resultado y = A*x:\n");
    for (int i = 0; i < n; i++) printf("%6.2f ", y_final[i]);
    printf("\n");
    free(M); free(V); free(y_final);
  }

  free(M_local); free(V_local); free(resultado_local);
  MPI_Finalize();
  return 0;
}

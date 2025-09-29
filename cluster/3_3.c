#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>   // Para gethostname


int main(int argc, char* argv[]){
  int rank, comm_sz;
  int valor_local; 
  int global_sum;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("Proceso %d de %d corriendo en %s\n", rank, comm_sz, hostname);

  srand(time(NULL));
  valor_local = ((int)rand() % 20 ) + 1;

  printf("Desde rank %d, con valor local %d \n", rank, valor_local);

  int suma = valor_local;



  int compa;
  for (int salto = 1; salto<comm_sz; salto *= 2){
    if(rank % (2*salto) == 0){ // receptor
      compa = rank + salto;
      if(compa < comm_sz){
        int recibe;
        MPI_Recv(&recibe, 1, MPI_INT, compa, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        suma += recibe;
      }
    }else{
      compa = rank - salto;
      MPI_Send(&suma, 1, MPI_INT, compa, 0, MPI_COMM_WORLD);
      break;
    }
  }

  if(rank == 0){
    global_sum = suma;
    printf("Suma global = %d\n", global_sum);
  }

  MPI_Finalize();
  return 0;

}

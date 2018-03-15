#include <iostream>
#include <chrono>
#include <mpi.h>
#include <random>
#include <math.h>
/*
 Author:
 Mitnyik Levente

 First Assistant:
 Captain Prof McBotos Csaba


*/

int main(int argc, char *argv[]){


  char hostname[MPI_MAX_PROCESSOR_NAME];
  int numtasks, rank, len, rc;


  //MPI_Status status;
  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Get_processor_name(hostname, &len);

  int N = 4;
  double* A = new double[N * N];
  double* B = new double[N * N];
  double* C = new double[N * N];
  int numGrid = sqrt(numtasks);
  int gridSize = N / numGrid;
  // what the actual fuck
  // if ((numGrid * numGrid) == numtasks){
  //   printf ("Use square number for the number of processes: %d != %d\n", numGrid*numGrid, numtasks);
  //   return 1;
  // }

  // initialize
  // for(int i=0; i<N; i++){
  //   for(int j=0; j<N; j++){
  //     A[i*N + j] = ((double) rand() / (RAND_MAX));
  //     B[i*N + j] = ((double) rand() / (RAND_MAX));
  //     C[i*N + j] = 0;
  //   }
  // }

  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      A[i*N + j] = 1.;
      B[i*N + j] = 1.;
      C[i*N + j] = 0.;
    }
  }
  //vau

  auto startTime = std::chrono::high_resolution_clock::now();

  for(int iter=0; iter<numGrid; iter++){
    for(int i=0; i<gridSize; i++){
      for(int j=0; j<gridSize; j++){
        for(int k=0; k<gridSize; k++){
          // Basic offset for the thread
          int offsetC = (rank / numGrid) * N * gridSize + rank % numGrid * gridSize;
          // Circular-shift column
          int offsetA = offsetC + ((rank + iter) % numGrid) * gridSize;
          // Circular-shift row
          int offsetB = offsetC + ((rank + iter) % numGrid) * N * gridSize;

          C[i*N + j + offsetC] = 1.;
          //std::cout << i*N + j + offsetC << std::endl;
          //C[i*N + j + offsetC] += A[i*N + k + offsetA] * B[k*N + j + offsetB];
        }
      }
    }
  }
  std::chrono::duration<double> endTime = std::chrono::high_resolution_clock::now() - startTime;
  std::cout << "rank " << rank << "  N:" << N << "  Time:" << endTime.count() << '\n';


  MPI_Finalize();
  if(rank == 3){
    for(int i=0; i<N; i++){
      for(int j=0; j<N; j++){

        printf("%2.2f  ", C[i*N + j]);
      }
      printf("\n");
    }
  }
}

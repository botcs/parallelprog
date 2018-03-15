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
  MPI_Status status;

  int N = 16;
  int numGrid = sqrt(numtasks);
  int gridSize = N / numGrid;

  double* A = new double[gridSize * gridSize];
  double* B = new double[gridSize * gridSize];
  double* C = new double[gridSize * gridSize];

  // initialize
  // for(int i=0; i<N; i++){
  //   for(int j=0; j<N; j++){
  //     A[i*N + j] = ((double) rand() / (RAND_MAX));
  //     B[i*N + j] = ((double) rand() / (RAND_MAX));
  //     C[i*N + j] = 0;
  //   }
  // }

  for(int i=0; i<gridSize; i++){
    for(int j=0; j<gridSize; j++){
      A[i*gridSize + j] = 1.;
      B[i*gridSize + j] = 1.;
      C[i*gridSize + j] = 0.;
    }
  }
  //vau

  auto startTime = std::chrono::high_resolution_clock::now();

  for(int iter=0; iter<numGrid; iter++){
    if(iter > 0){

    }
    for(int i=0; i<gridSize; i++){
      for(int j=0; j<gridSize; j++){
        for(int k=0; k<gridSize; k++){
          C[i*gridSize + j] += A[i*gridSize + k] * B[k*gridSize + j];
        }
      }
    }

    // Start COL shifting
    if(rank % numGrid == 0){
      // Cells in the first COLUMN are initializing the shift

      //           FIRST ROW ELEMENT    +  MODULO OFFSET
      int toRank = rank/numGrid*numGrid + (numGrid + rank-1)%numGrid;

      std::cout << ">>>SEND>>> A block my id " << rank << "  TO:" << toRank << '\n';
      MPI_Send(A, gridSize*gridSize,MPI_DOUBLE,toRank,22,MPI_COMM_WORLD);

      //             FIRST ROW ELEMENT    +  MODULO OFFSET
      int fromRank = rank/numGrid*numGrid + (rank+1)%numGrid;
      std::cout << "<<<REC<<< A block my id " << rank << "  FROM:" << fromRank << '\n';
      MPI_Recv(A,gridSize*gridSize,MPI_DOUBLE,fromRank,22,MPI_COMM_WORLD,&status);
    } else {
      // Cells in the other parts are now waiting for the MPI_Send
      // instead of waiting in deadlock

      //             FIRST ROW ELEMENT    +  MODULO OFFSET
      int fromRank = rank/numGrid*numGrid + (rank+1)%numGrid;
      std::cout << "<<<REC<<< A block my id " << rank << "  FROM:" << fromRank << '\n';
      MPI_Recv(A,gridSize*gridSize,MPI_DOUBLE,fromRank,22,MPI_COMM_WORLD,&status);

      //           FIRST ROW ELEMENT    +  MODULO OFFSET
      int toRank = rank/numGrid*numGrid + (numGrid + rank-1)%numGrid;

      std::cout << ">>>SEND>>> A block my id " << rank << "  TO:" << toRank << '\n';
      MPI_Send(A, gridSize*gridSize,MPI_DOUBLE,toRank,22,MPI_COMM_WORLD);
    }

    // Start ROW shifting
    if(rank < numGrid){
      // Cells in the first ROW are initializing the shift
      std::cout << ">>>SEND>>> B block my id " << rank << "  TO:" << (numtasks + rank-numGrid)%numtasks << '\n';
      MPI_Send(B, gridSize*gridSize,MPI_DOUBLE,(numtasks + rank-numGrid)%numtasks,22,MPI_COMM_WORLD);

      std::cout << "<<<REC<<< B block my id " << rank << "  FROM:" << (rank+numGrid)%numtasks << '\n';
      MPI_Recv(B,gridSize*gridSize,MPI_DOUBLE,(rank+numGrid)%numtasks,22,MPI_COMM_WORLD,&status);
    } else {
      // Cells in the other parts are now waiting for the MPI_Send
      // instead of waiting in deadlock
      std::cout << "<<<REC<<< B block my id " << rank << "  FROM:" << (rank+numGrid)%numtasks << '\n';
      MPI_Recv(B,gridSize*gridSize,MPI_DOUBLE,(rank+numGrid)%numtasks,22,MPI_COMM_WORLD,&status);

      std::cout << ">>>SEND>>> B block my id " << rank << "  TO:" << (numtasks + rank-numGrid)%numtasks << '\n';
      MPI_Send(B, gridSize*gridSize,MPI_DOUBLE,(numtasks + rank-numGrid)%numtasks,22,MPI_COMM_WORLD);
    }
  }
  if(rank == 0){
    double* C_all = new double[gridSize * gridSize];
  }

  std::chrono::duration<double> endTime = std::chrono::high_resolution_clock::now() - startTime;
  std::cout << "Worker finished" << rank << "  N:" << N << "  Time:" << endTime.count() << '\n';



  MPI_Finalize();
  if(rank == 3){
    for(int i=0; i<gridSize; i++){
      for(int j=0; j<gridSize; j++){

        printf("%2.2f  ", C[i*gridSize + j]);
      }
      printf("\n");
    }
  }
}

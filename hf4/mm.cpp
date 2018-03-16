#include <iostream>
#include <chrono>
#include <mpi.h>
#include <random>
#include <math.h>

//#define VERBOSE true


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

  int N = 4;
  int numGrid = sqrt(numtasks);
  int gridSize = N / numGrid;

  double* A = new double[gridSize * gridSize];
  double* B = new double[gridSize * gridSize];
  double* C = new double[gridSize * gridSize];

  // Required for gathering, buffering subprocess matrices
  double* C_all;
  double* buffA = new double[gridSize * gridSize];
  double* buffB = new double[gridSize * gridSize];

  // initialize
  // for(int i=0; i<N; i++){
  //   for(int j=0; j<N; j++){
  //     A[i*N + j] = ((double) rand() / (RAND_MAX));
  //     B[i*N + j] = ((double) rand() / (RAND_MAX));
  //     C[i*N + j] = 0;
  //   }
  // }

  // for(int i=0; i<gridSize; i++){
  //   for(int j=0; j<gridSize; j++){
  //     A[i*gridSize + j] = 1.;
  //     B[i*gridSize + j] = 1.;
  //     C[i*gridSize + j] = 0.;
  //   }
  // }
  //vau

  // Results in tiled diagonal...
  // for(int i=0; i<gridSize; i++){
  //   for(int j=0; j<gridSize; j++){
  //     if(i==j){
  //       A[i*gridSize + j] = 1.;
  //       B[i*gridSize + j] = 1.;
  //     } else {
  //       A[i*gridSize + j] = 0.;
  //       B[i*gridSize + j] = 0.;
  //     }
  //     C[i*gridSize + j] = 0.;
  //   }
  // }

  // //Initialize a UNIT matrix
  // for(int i=0; i<gridSize; i++){
  //   for(int j=0; j<gridSize; j++){
  //     if(i==j && rank % (numGrid + 1) == 0){
  //       A[i*gridSize + j] = 1.;
  //       B[i*gridSize + j] = 1.;
  //     } else {
  //       A[i*gridSize + j] = 0.;
  //       B[i*gridSize + j] = 0.;
  //     }
  //     C[i*gridSize + j] = 0.;
  //   }
  // }


  for(int i=0; i<gridSize; i++){
    for(int j=0; j<gridSize; j++){
      int offset = (rank / numGrid) * N * gridSize + rank % numGrid * gridSize;
      A[i*gridSize + j] = double(i*N + j + offset);
      B[i*gridSize + j] = double(i*N + j + offset);

      C[i*gridSize + j] = 0.;
    }
  }
  auto startTime = std::chrono::high_resolution_clock::now();

  for(int iter=0; iter<numGrid; iter++){


    // COLUMN shifting
    if(iter>0){
      if(rank % numGrid == 0){
        // Cells in the first COLUMN are initializing the shift

        //           FIRST ROW ELEMENT    +  MODULO OFFSET
        int toRank = rank/numGrid*numGrid + (numGrid + rank-1)%numGrid;

        #ifdef VERBOSE
          std::cout << ">>>SEND>>> A block my id " << rank << "  TO:" << toRank << '\n';
        #endif
        MPI_Send(A, gridSize*gridSize,MPI_DOUBLE,toRank,22,MPI_COMM_WORLD);

        //             FIRST ROW ELEMENT    +  MODULO OFFSET
        int fromRank = rank/numGrid*numGrid + (rank+1)%numGrid;
        #ifdef VERBOSE
          std::cout << "<<<REC<<< A block my id " << rank << "  FROM:" << fromRank << '\n';
        #endif
        MPI_Recv(A,gridSize*gridSize,MPI_DOUBLE,fromRank,22,MPI_COMM_WORLD,&status);
      } else {
        // Cells in the other parts are now waiting for the MPI_Send
        // instead of waiting in deadlock

        //             FIRST ROW ELEMENT    +  MODULO OFFSET
        int fromRank = rank/numGrid*numGrid + (rank+1)%numGrid;
        #ifdef VERBOSE
          std::cout << "<<<REC<<< A block my id " << rank << "  FROM:" << fromRank << '\n';
        #endif
        MPI_Recv(buffA,gridSize*gridSize,MPI_DOUBLE,fromRank,22,MPI_COMM_WORLD,&status);

        //           FIRST ROW ELEMENT    +  MODULO OFFSET
        int toRank = rank/numGrid*numGrid + (numGrid + rank-1)%numGrid;

        #ifdef VERBOSE
          std::cout << ">>>SEND>>> A block my id " << rank << "  TO:" << toRank << '\n';
        #endif
        MPI_Send(A, gridSize*gridSize,MPI_DOUBLE,toRank,22,MPI_COMM_WORLD);

        // Assign buffA only after A was sent
        for(int i=0; i<gridSize; i++){
          for(int j=0; j<gridSize; j++){
            A[i*gridSize + j] = buffA[i*gridSize + j];
          }
        }
      }

      // ROW shifting
      if(rank < numGrid){
        // Cells in the first ROW are initializing the shift
        #ifdef VERBOSE
          std::cout << ">>>SEND>>> B block my id " << rank << "  TO:" << (numtasks + rank-numGrid)%numtasks << '\n';
        #endif
        MPI_Send(B, gridSize*gridSize,MPI_DOUBLE,(numtasks + rank-numGrid)%numtasks,22,MPI_COMM_WORLD);

        #ifdef VERBOSE
          std::cout << "<<<REC<<< B block my id " << rank << "  FROM:" << (rank+numGrid)%numtasks << '\n';
        #endif
        MPI_Recv(B,gridSize*gridSize,MPI_DOUBLE,(rank+numGrid)%numtasks,22,MPI_COMM_WORLD,&status);
      } else {
        // Cells in the other parts are now waiting for the MPI_Send
        // instead of waiting in deadlock
        #ifdef VERBOSE
          std::cout << "<<<REC<<< B block my id " << rank << "  FROM:" << (rank+numGrid)%numtasks << '\n';
        #endif
        MPI_Recv(buffB,gridSize*gridSize,MPI_DOUBLE,(rank+numGrid)%numtasks,22,MPI_COMM_WORLD,&status);

        #ifdef VERBOSE
          std::cout << ">>>SEND>>> B block my id " << rank << "  TO:" << (numtasks + rank-numGrid)%numtasks << '\n';
        #endif
        MPI_Send(B, gridSize*gridSize,MPI_DOUBLE,(numtasks + rank-numGrid)%numtasks,22,MPI_COMM_WORLD);

        // Assign buffB only after B was sent
        for(int i=0; i<gridSize; i++){
          for(int j=0; j<gridSize; j++){
            B[i*gridSize + j] = buffB[i*gridSize + j];
          }
        }
      }
    }
    for(int i=0; i<gridSize; i++){
      for(int j=0; j<gridSize; j++){
        for(int k=0; k<gridSize; k++){
          C[i*gridSize + j] += A[i*gridSize + k] * B[k*gridSize + j];
        }
      }
    }
    #ifdef VERBOSE
      if(rank == 0){
        std::cout << "MASTER A\n";
        for(int i=0; i<gridSize; i++){
          for(int j=0; j<gridSize; j++){
            printf("%2.0f  ", A[i*gridSize + j]);
          }
          printf("\n");
        }

        std::cout << "MASTER B\n";
        for(int i=0; i<gridSize; i++){
          for(int j=0; j<gridSize; j++){
            printf("%2.0f  ", B[i*gridSize + j]);
          }
          printf("\n");
        }

        std::cout << "MASTER C\n";
        for(int i=0; i<gridSize; i++){
          for(int j=0; j<gridSize; j++){
            printf("%2.0f  ", C[i*gridSize + j]);
          }
          printf("\n");
        }
      }
    #endif
  }
  if(rank == 0){
    C_all = new double[N * N];
    for(int i=0; i<N; i++){
      for(int j=0; j<N; j++){
        C_all[i, j] = 0;
      }
    }
    // MPI gather would be handy
    for(int fromRank=0; fromRank<numtasks; fromRank++){
      if(fromRank > 0){
        std::cout << "+++FINAL REC<<< C block MASTER FROM " << fromRank << '\n';
        MPI_Recv(C,gridSize*gridSize,MPI_DOUBLE,fromRank,22,MPI_COMM_WORLD,&status);
      }
      int offsetC = (fromRank / numGrid) * N * gridSize + fromRank % numGrid * gridSize;
      for(int i=0; i<gridSize; i++){
        for(int j=0; j<gridSize; j++){
          C_all[i*N + j + offsetC] = C[i*gridSize + j];
        }
      }
    }
    std::chrono::duration<double> endTime = std::chrono::high_resolution_clock::now() - startTime;
    std::cout << "MASTER finished: " << rank << "  Time:" << endTime.count() << '\n';
  } else {
    std::cout << "+++FINAL SEND>>> C block, my id " << rank << "  TO: MASTER THREAD\n";
    MPI_Send(C, gridSize*gridSize,MPI_DOUBLE,0,22,MPI_COMM_WORLD);
    std::chrono::duration<double> endTime = std::chrono::high_resolution_clock::now() - startTime;
    std::cout << "Worker finished: " << rank << "  Time:" << endTime.count() << '\n';
  }




  MPI_Finalize();
  if(rank == 0){
    for(int i=0; i<N; i++){
      for(int j=0; j<N; j++){
        printf("%2.0f  ", C_all[i*N + j]);
      }
      printf("\n");
    }
    delete C_all;
  }
  delete A, B, C;
}

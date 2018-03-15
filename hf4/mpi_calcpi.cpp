#include <random>
#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
  int numtasks, rank, len, rc;

  int npoints = 10000000;
  int circle_count = 0;

  char hostname[MPI_MAX_PROCESSOR_NAME];

  //MPI_Status status;
  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Get_processor_name(hostname, &len);


  if (rank == 0) {
    printf("MASTER task is started\n");
  } else {
    printf("WORKER task is started, worker ID: %d\n", rank);
  }

  int p = numtasks;
  int num = npoints / p;

  //This is how we generate random numbers with C++11
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-0.5, 0.5);
  for (int i=0; i<num; i++){
    double x = dis(gen);
    double y = dis(gen);
    if ((x * x + y * y) <= 0.25){
      circle_count++;
    }
  }


  int buffer;
  int rbuffer;
  MPI_Status status;
  // if I am MASTER
  if (rank == 0) {
    for (int i=1; i<numtasks; i++){
      MPI_Recv(&rbuffer,1,MPI_INT,i,22,MPI_COMM_WORLD,&status);
      circle_count += rbuffer;
    }
    double pi = 4.0 * (circle_count / double(npoints));
    printf("MASTER task is finished, approx of PI is: %1.4f\n", pi);

  } else {
    buffer = circle_count;
    MPI_Send(&buffer,1,MPI_INT,0,22,MPI_COMM_WORLD);
    printf("WORKER task is finished, worker ID: %d\n", rank);
  }
  //printf ("Number of tasks= %d My rank= %d Running on %s\n",
  //  numtasks,rank,hostname);

  MPI_Finalize();
}

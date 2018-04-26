#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int my_rank;
int nprocs;
int nprocs_y;
int nprocs_x;
int my_rank_x;
int my_rank_y;
int prev_y;
int next_y;
int next_x;
int prev_x;
MPI_Datatype vertSlice, horizSlice;
int imax_full;
int jmax_full;
int gbl_i_begin;
int gbl_j_begin;

double* dat_ptrs[6];
std::map<double*, bool> dat_dirty;

void mpi_setup(int argc, char **argv, int *imax, int *jmax) {
  //Initialise: get #of processes and process id
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  //Check for compatible number of processes
  if(my_rank==0)
    printf("nprocs: %d\n", nprocs);
  if ((int)(sqrt(nprocs))*(int)(sqrt(nprocs)) != nprocs) {
    printf("Error, not a square number of processes!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
	
  //Figure out process X,Y coordinates
  nprocs_x = sqrt(nprocs);
  nprocs_y = sqrt(nprocs);
  
  my_rank_x = my_rank % nprocs_x;
  my_rank_y = my_rank / nprocs_x;
  //my_rank = my_rank_y*nprocs_x+my_rank_x;

  //Figure out neighbours
  prev_x = (my_rank_x-1)<0 ? MPI_PROC_NULL : my_rank-1;
  next_x = (my_rank_x+1)>=nprocs_x ? MPI_PROC_NULL : my_rank+1;

  prev_y = (my_rank_y-1)<0 ? MPI_PROC_NULL : my_rank-nprocs_x;
  next_y = (my_rank_y+1)>=nprocs_y ? MPI_PROC_NULL : my_rank+nprocs_x;
  
  //Save original full sizes in x and y directions
  imax_full = *imax;
  jmax_full = *jmax;
	
  //Modify imax and jmax (pay attention to integer divisions's rounding issues!)
  *imax = (my_rank_x != nprocs_x-1) ? imax_full/nprocs_x : imax_full - my_rank_x * (imax_full/nprocs_x);
  *jmax = (my_rank_y != nprocs_y-1) ? jmax_full/nprocs_y : jmax_full - my_rank_y * (jmax_full/nprocs_y);

  //Figure out beginning i and j index in terms of global indexing
  gbl_i_begin = my_rank_x * (imax_full/nprocs_x);   
  gbl_j_begin = my_rank_y * (jmax_full/nprocs_y);   

  //Let's set up MPI Datatypes
  //Homework: ghost cells are not 1 on each side, but 2! Change these to send 2 rows/columns at the same time
  MPI_Type_vector((*jmax)*2+8,2,(*imax)+1, MPI_DOUBLE, &vertSlice); 
  MPI_Type_vector((*imax)*2+8,1,1, MPI_DOUBLE, &horizSlice);
  MPI_Type_commit(&vertSlice);
  MPI_Type_commit(&horizSlice); 
	
}

void exchange_halo(int imax, int jmax, double *arr) {
  //std::cout << "my rank: "<< my_rank << " my neighbours: "<< prev_x << next_x << prev_y << next_y;
 
  //std::cout << "dirty";
  
  auto registered = dat_dirty.find(arr);
  if (registered == dat_dirty.end()){
    std::cout << "Array has not been registered";
    exit(1);
  }
  bool dirty = dat_dirty[arr];
  if (dirty) {
    //Homework: ghost cells are not 1 on each side, but 2!
    // since we are sending 2 rows/columns, make sure the offsets into arr are right!
    //Exchange halos: top, bottom, left, right

//    std::cout << "my rank " << my_rank << " neighbours" << prev_x <<  " " << prev_y << " " << next_x <<  " " << next_y << "\n";

    //send my REAL LEFT edge to prev_x, receive prev_x RIGHT edge to my GHOST LEFT edge
    MPI_Request requests[8];
    int counter = 0;
    if(prev_x != MPI_PROC_NULL){
      MPI_Isend(&arr[0*(imax+4)+2],1,vertSlice,prev_x,0,MPI_COMM_WORLD,&requests[counter++]);
      MPI_Irecv(&arr[0*(imax+4)+0],1,vertSlice,prev_x,0,MPI_COMM_WORLD,&requests[counter++]);
      //MPI_Sendrecv(&arr[0*(imax+4)+(imax)],1,MPI_DOUBLE,prev_x, 0,
      //             &arr[0*(imax+4)+0]     ,1,MPI_DOUBLE,my_rank,0,
      //             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    //send my REAL RIGHT edge to next_x, receive next_x LEFT edge to my GHOST RIGHT edge 
    if(next_x != MPI_PROC_NULL){
      MPI_Isend(&arr[0*(imax+4)+jmax],  1,vertSlice,next_x,0,MPI_COMM_WORLD,&requests[counter++]);
      MPI_Irecv(&arr[0*(imax+4)+imax+2],1,vertSlice,next_x,0,MPI_COMM_WORLD,&requests[counter++]);
      //MPI_Sendrecv(&arr[0*(imax+4)+2],     1,vertSlice,prev_x,0,
      //             &arr[0*(imax+4)+imax+2],1,vertSlice,next_x,0,
      //             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    //send my REAL TOP edge to prev_y, receive prev_y BOTTOM edge to my GHOST TOP edge
    if(prev_y != MPI_PROC_NULL){
      MPI_Isend(&arr[2*(imax+4)+0],1,horizSlice,prev_y,0,MPI_COMM_WORLD,&requests[counter++]);
      MPI_Irecv(&arr[0*(imax+4)+0],1,horizSlice,prev_y,0,MPI_COMM_WORLD,&requests[counter++]);
      //MPI_Sendrecv(&arr[(jmax)*(imax+4)+0],1,horizSlice,next_y,0,
      //             &arr[0*(imax+4)+0]     ,1,horizSlice,my_rank,0,
      //             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    //send my REAL BOTTOM edge to next_y, receive next_y TOP edge to my GHOST BOTTOM edge
    if(next_y != MPI_PROC_NULL){
      MPI_Isend(&arr[(jmax)*(imax+4)+0],  1,horizSlice,next_y,0,MPI_COMM_WORLD,&requests[counter++]);
      MPI_Irecv(&arr[(jmax+2)*(imax+4)+0],1,horizSlice,next_y,0,MPI_COMM_WORLD,&requests[counter++]);
      //MPI_Sendrecv(&arr[2*(imax+4)+0],       1,horizSlice,prev_y,0,
      //             &arr[(jmax+2)*(imax+4)+0],1,horizSlice,next_y,0,
      //             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    //printf("%d %d %d %d %d %d\n", my_rank,counter, requests[0], requests[1], requests[2], requests[3]);
    dat_dirty[arr] = false;
  }
}

void set_dirty(double *arr) {
  dat_dirty[arr] = true;
}

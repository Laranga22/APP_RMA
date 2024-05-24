/*
 * Example of active target synchronization using PSCW
 * 
 * Processes divide in 2 groups. Odd processes send data to even processes
 * 
 * Compile: mpicc -Wall -O3 -std=c99 -o 02_rma_pscw 02_rma_pscw.c
 * Run: no arguments required
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define A_SIZE 1000

int main(int argc, char ** argv)
{
  int *my_array;
  int size, rank;
  MPI_Group orig_group, even_group, odd_group;
  MPI_Win win;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //TODO: Create a Group out of the communicator with MPI_Comm_group
  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

  //TODO: Include the odd/even processes in the group
  int *ranks = (int *)malloc(size/2 * sizeof(int));
  for (int i = 0; i < size; i += 2) {
    ranks[i/2] = i;
  }
  MPI_Group_incl(orig_group, size/2, ranks, &even_group);
  for(int i = 1; i < size; i += 2) {
    ranks[i/2] = i;
  }
  MPI_Group_incl(orig_group, (size+1)/2, ranks, &odd_group);
  free(ranks);
  
  /* create private memory */
  //TODO: Create a memory window in "my_array",
  //      allocating memory at the same time
  MPI_Alloc_mem(size * sizeof(int), MPI_INFO_NULL, &my_array);
  MPI_Win_create(my_array, size * sizeof(int), sizeof(int),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  if (rank % 2)
  {
    /* odd processes send data: 
     * we will place our rank in our partners' memory */
    int send_data = rank;
    
    //TODO: Enclose operations within an access epoch to the partner group
    MPI_Win_start(even_group, 0, win);
    //TODO: Replace these Two-Sided with One-Sided operations
    for (int i=0; i<size; i+=2)
      MPI_Put(&send_data, 1, MPI_INT, i, rank, 1, MPI_INT, win);
    
    MPI_Win_complete(win);
  }
  else
  {
    /* even processes receive data */
    
    //TODO: Create an exposure epoch to the partner group
    MPI_Win_post(odd_group, 0, win);

    //TODO: No operation call is needed in the target part
    MPI_Win_wait(win);
  }

  for (int i=0; i<size; i++)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == rank)
    {
      printf("Process %2d. A={ ", rank);
      for (int j=0; j<size; j++)
        if (my_array[j])
          printf("%2d ", my_array[j]);
        else
          printf(" - ");
      printf(" }\n");
    }
  }
  
  //TODO: Free MPI Window
  MPI_Win_free(&win);
  MPI_Free_mem(my_array);
    
  MPI_Finalize();
  return 0;
}





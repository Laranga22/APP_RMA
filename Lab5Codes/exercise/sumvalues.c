#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define ERR_ARGS 1
#define ERR_IO   2
#define ERR_MEM  3
#define ERR_READ 4

/* custom variable to use instead of MPI_UNIVERSE_SIZE for testing purposes */
int CUSTOM_UNIVERSE_SIZE;
#define CUSTOM_UNIVERSE_SIZE_VALUE 8

void __rsv_set_comm_attr();
int main( int argc, char *argv[])
{
  int mpi_rank, mpi_size;
  int gsize[2];
  int lsize[2];
  int *mat;
  FILE * input_file;

  int *displs=0, *send_counts=0;
   
  MPI_Init(&argc, &argv);

  __rsv_set_comm_attr();

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  //TODO: Obtain the custom universe size from the communicater attribute CUSTOM_UNIVERSE_SIZE

  if(argc < 4)
  {
    if (!mpi_rank)
    {
      printf("Usage: %s file rows cols\n\n", argv[0]);
    }
    MPI_Finalize();
    return ERR_ARGS;
  }

  gsize[0] = atoi(argv[2]);
  gsize[1] = atoi(argv[3]);
  
  //TODO: Decide the number of processes to use based on the matriz size (gsize)

  lsize[0] = gsize[0]/mpi_size + (mpi_rank < gsize[0]%mpi_size?1:0);
  lsize[1] = gsize[1]; 

  //TODO: Modify the algorithm for working in shared memory with MPI
  
  if (!mpi_rank)
  {
    if (!(input_file = fopen(argv[1], "r")))
    {
      fprintf(stderr, "Error opening input file or dimensions are wrong\n");
      MPI_Abort(MPI_COMM_WORLD, ERR_IO);
    }
    if (!(mat = (int *) malloc (sizeof(int)*gsize[0]*gsize[1])))
    {
      fprintf(stderr, "Error allocating memory\n");
      MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
    }
    if (fread(mat, sizeof(int), gsize[0]*gsize[1], input_file) != gsize[0]*gsize[1])
    {
      fprintf(stderr, "Error reading input file or dimensions are wrong\n");
      MPI_Abort(MPI_COMM_WORLD, ERR_READ);
    }
    
    displs      = (int *) malloc (mpi_size * sizeof(int));
    send_counts = (int *) malloc (mpi_size * sizeof(int));
    displs[0] = 0;
    send_counts[0] = lsize[0]*lsize[1];
    for (int i=1; i<mpi_size; i++)
    {
      displs[i] = displs[i-1] + send_counts[i-1];
      send_counts[i] = (gsize[0]/mpi_size + (i < gsize[0]%mpi_size?1:0)) * lsize[1];
    }
  }
  else
  {
    if (!(mat = (int *) malloc (sizeof(int)*lsize[0]*lsize[1])))
    {
      fprintf(stderr, "Error allocating memory\n");
      MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
    }
  }
  
  MPI_Scatterv(mat, send_counts, displs, MPI_INT,
               mpi_rank?mat:MPI_IN_PLACE, lsize[0]*lsize[1], MPI_INT,
               0, MPI_COMM_WORLD);
 
  /* sum */
  int sum = 0;
  for (int i=0; i<lsize[0]; ++i)
    for (int j=0; j<lsize[1]; ++j)
      sum += mat[i*lsize[1] + j];
  
  MPI_Reduce(mpi_rank?&sum:MPI_IN_PLACE, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (!mpi_rank)
    printf("Sum is %d\n", sum);
   
  MPI_Finalize();
  
  free(mat);

  return 0;
}


void __rsv_set_comm_attr()
{
  #define CUSTOM_UNIVERSE_SIZE       usk

  int usv = CUSTOM_UNIVERSE_SIZE_VALUE;

  MPI_Comm_create_keyval( MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN,
                          &CUSTOM_UNIVERSE_SIZE, (void *)0 );
  MPI_Comm_set_attr( MPI_COMM_WORLD, CUSTOM_UNIVERSE_SIZE, &usv );
}
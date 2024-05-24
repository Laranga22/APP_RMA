/*
 * IO example: Simple centralized I/O
 *
 * This example:
 * 1. Reads an array of integers from a file
 * 2. Scatters it among all processes
 * 4. Gathers the array back
 * 5. Prints the results to an output file
 *
 * Note that, input and output files have a binary format!
 *
 * Compile: mpicc -Wall -o io_01_readwrite_1D io_01_readwrite_1D.c
 * Run arguments: input_file output_file
 * e.g: mpirun -n N ./io_01_readwrite_1D data/integers.input data/integers.output
 * 
 * Disclaimer: For simplicity, many error checking statements
 *             are omitted.
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

#define V_PER_ROW 16
#define DO_PRINT   0

int f(int v)
{
  return v + 1;
}

void print_array(int * m, int w)
{
  /* print */
  for (int x=0; x<w; ++x)
  {
      printf("%7d ", m[x]);
      if (x && !(x % V_PER_ROW))
        printf("\n");
  }

  if (w % V_PER_ROW)
    printf("\n");

  fflush(stdout);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  /* mpi_defs */
  int mpi_rank, mpi_size;

  /* other stuff */
  int *array = 0,   /* global array */
      *l_array = 0; /* local array */

  char * input_filename  = argv[1];
  char * output_filename = argv[2];
  int gsize;
  int lsize;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc != 3)
  {
    if (!mpi_rank)
      printf("Usage: %s input_fname output_fname\n", argv[0]);
    MPI_Finalize();
    exit(1);
  }

  //TODO: Replace step 1 with a collective File open.

  /* 1. Root process reads the input file */
  MPI_File infile, outfile;
  MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);

  MPI_Offset file_size;
  MPI_File_get_size(infile, &file_size);

  gsize = file_size / sizeof(int);
  lsize = gsize / mpi_size;

  l_array = (int *) malloc(lsize * sizeof(int));

  MPI_File_read_at_all(infile, mpi_rank * lsize * sizeof(int), l_array, lsize, MPI_INT, MPI_STATUS_IGNORE);

  MPI_File_close(&infile);

#if(DO_PRINT)
  for (int p=0; p<mpi_size; ++p)
  {
    if (mpi_rank == p)
    {
      printf("\nProcess %d/%d, local size =  %d\n",
             mpi_rank, mpi_size, lsize);
      print_array(l_array, lsize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  //TODO: Replace steps 3 and 4 with MPI I/O operations (analogous to 1 and 2)
  //      but now, each process will define its own File View to perform a
  //      collective write.

  MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);

  MPI_File_write_at_all(outfile, mpi_rank * lsize * sizeof(int), l_array, lsize, MPI_INT, MPI_STATUS_IGNORE);

  MPI_File_close(&outfile);

  free(l_array);
  if (!mpi_rank)
    free(array);

  MPI_Finalize();

  return 0;
}
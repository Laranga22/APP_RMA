/*
 * Exercise of one-sided operations with passive target synchronization
 * 
 * Each process generates a list of random values and computes
 * a set of basic statistical metrics:
 * minimum, maximum, mean, and standard deviation
 * 
 * Compile: mpicc -Wall -o rma_03_passive rma_03_passive.c -lm
 * Run: no arguments required
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define A_SIZE 1000000000
#define V_MAX 100.0

typedef struct 
{
  double min;
  double max;
  double average;
  double standard_deviation;
} stats_t;

int main(int argc, char **argv)
{
  int mpi_size, mpi_rank;
  double *my_array;
  stats_t stats;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int block_size = A_SIZE / mpi_size;

  /* generate a set of random values */
  my_array = (double *)malloc(block_size * sizeof(double));
  srand(12345 + mpi_rank);
  for (int i = 0; i < block_size; i++)
    my_array[i] = (1.0 * rand() / INT_MAX) * V_MAX;

  /* initialize stats */
  stats.min = stats.max = my_array[0];
  stats.average = 0;
  double sumvalues = 0;
  double sumsquared = 0;

  /* compute stats */
  for (int i = 0; i < block_size; i++)
  {
    if (my_array[i] < stats.min)
      stats.min = my_array[i];
    if (my_array[i] > stats.max)
      stats.max = my_array[i];
    sumvalues += my_array[i];
    sumsquared += my_array[i] * my_array[i];
  }
  stats.average = sumvalues / block_size;
  stats.standard_deviation =
    sqrt(
      (sumsquared + 2*sumvalues*stats.average + stats.average*stats.average)
      / block_size);
  // Create window for global_stats
  stats_t global_stats;
  if (mpi_rank == 0)
  {
    global_stats.min = V_MAX;
    global_stats.max = -V_MAX;
    global_stats.average = 0;
    global_stats.standard_deviation = 0;
  }
  MPI_Win win_global_stats;
  MPI_Win_create(mpi_rank == 0 ? &global_stats : MPI_BOTTOM, mpi_rank == 0 ? sizeof(stats_t) : 0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_global_stats);

  // Accumulate stats to global_stats
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_global_stats);
  MPI_Accumulate(&stats.min, 1, MPI_DOUBLE, 0, offsetof(stats_t, min) / sizeof(double), 1, MPI_DOUBLE, MPI_MIN, win_global_stats);
  MPI_Accumulate(&stats.max, 1, MPI_DOUBLE, 0, offsetof(stats_t, max) / sizeof(double), 1, MPI_DOUBLE, MPI_MAX, win_global_stats);
  MPI_Accumulate(&stats.average, 1, MPI_DOUBLE, 0, offsetof(stats_t, average) / sizeof(double), 1, MPI_DOUBLE, MPI_SUM, win_global_stats);
  MPI_Accumulate(&stats.standard_deviation, 1, MPI_DOUBLE, 0, offsetof(stats_t, standard_deviation) / sizeof(double), 1, MPI_DOUBLE, MPI_SUM, win_global_stats);
  MPI_Win_unlock(0, win_global_stats);

  MPI_Win_free(&win_global_stats);

  if (mpi_rank == 0)
  {
    global_stats.average /= mpi_size;
    global_stats.standard_deviation /= mpi_size;

    printf("     %8.4lf %8.4lf %8.4lf %10.4lf\n",
           global_stats.min,
           global_stats.max,
           global_stats.average,
           global_stats.standard_deviation);
  }

  // Create window for all_stats
  stats_t *all_stats = NULL;
  if (mpi_rank == 0)
  {
    all_stats = (stats_t *)malloc(mpi_size * sizeof(stats_t));
  }
  MPI_Win win_all_stats;
  MPI_Win_create(mpi_rank == 0 ? all_stats : MPI_BOTTOM, mpi_rank == 0 ? mpi_size * sizeof(stats_t) : 0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_all_stats);

  // Gather individual stats to all_stats
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_all_stats);
  MPI_Put(&stats, sizeof(stats_t) / sizeof(double), MPI_DOUBLE, 0, mpi_rank * (sizeof(stats_t) / sizeof(double)), sizeof(stats_t) / sizeof(double), MPI_DOUBLE, win_all_stats);
  MPI_Win_unlock(0, win_all_stats);

  MPI_Win_free(&win_all_stats);

  if (mpi_rank == 0)
  {
    /* print individual stats */
    printf("  proc   minimum  maximum    mean  standard_dev\n");
    printf("-----------------------------------------------\n");
    for (int i = 0; i < mpi_size; i++)
    {
      printf("  %3d   %8.4lf %8.4lf %8.4lf %10.4lf\n",
             i,
             all_stats[i].min,
             all_stats[i].max,
             all_stats[i].average,
             all_stats[i].standard_deviation);
    }
    printf("-----------------------------------------------\n");
    free(all_stats);
  }

  free(my_array);
  MPI_Finalize();
  return 0;
}
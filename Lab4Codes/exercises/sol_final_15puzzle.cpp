/*
 * Exercise of dynamic load balancing
 *
 * This program is a 15-puzzle solver (google 15-puzzle if you don't know what it is)
 * Each 15-puzzle initial state, a set of 15 tiles placed into a 4-by-4 grid, is encoded as a "size_t" hash
 * The program reads a file with a set of hashes and attempts to solve the puzzles.
 * Note that not every initial configuration is solvable.
 * 
 * The goal of this exercise is to modify this source code to use MPI-IO and one-sided operations.
 * Analyze the workload carefully and reason the best solution in order to improve the performance
 * for every potential configuration.
 *
 * This code depends on aux_files/15_puzzle.{h, cpp} which contains the actual solver.
 * The optimality of the solver can be configured when calling the solve() function in line 112.
 * Try there different values to check if the workload can be balanced correctly. 
 * 
 * Compile: mpic++ -O3 -o final_15puzzle final_15puzzle.cpp aux_files/15puzzle.cpp -lm
 * Run: mpirun -n NP ./final_15puzzle data/puzzles.100.bin
 */

#include "aux_files/15puzzle.h"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;
using namespace std::chrono;

#define PUZZLE_COUNT 1000
#define DEBUG_LEVEL 2

int main(int argc, char *argv[])
{
    int mpi_rank, mpi_size;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        cerr << "Error initializing MPI" << endl;
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    MPI_File input_fp;
    size_t *puzzles = nullptr;
    int puzzle_count = 0;
    MPI_Offset file_size;

    if (MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_fp) != MPI_SUCCESS)
    {
        cerr << "Cannot read input file " << argv[1] << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_File_get_size(input_fp, &file_size);
    puzzle_count = file_size / sizeof(size_t);

    if (mpi_rank == 0)
    {
        puzzles = (size_t *)malloc(sizeof(size_t) * puzzle_count);
        MPI_File_read(input_fp, puzzles, puzzle_count, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&input_fp);

    MPI_Win puzzle_win;
    MPI_Win_create(puzzles, (mpi_rank == 0) ? puzzle_count * sizeof(size_t) : 0, sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD, &puzzle_win);

    initialize_rng(42 + mpi_rank);

    auto ps_time = high_resolution_clock::now();
    int task_index = mpi_rank;

    vector<double> solve_times;

    while (task_index < puzzle_count)
    {
        size_t current_puzzle;
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, puzzle_win);
        MPI_Get(&current_puzzle, 1, MPI_UNSIGNED_LONG, 0, task_index, 1, MPI_UNSIGNED_LONG, puzzle_win);
        MPI_Win_unlock(0, puzzle_win);

        auto s_time = high_resolution_clock::now();
        auto hash_state = new_from_hash(current_puzzle);

        char solution_moves[MAX_SOLUTION_LENGTH];
        int path_length, nodes_alloc;
        int can_solve = solve(hash_state, &path_length, &nodes_alloc, OPT_MEH, solution_moves);
        auto e_time = high_resolution_clock::now();
        auto local_duration = duration_cast<microseconds>(e_time - s_time);

        solve_times.push_back(local_duration.count() / 1000.0);

#if (DEBUG_LEVEL > 0)
        cout << "[" << mpi_rank << ":" << setfill('0') << setw(ceil(log10(puzzle_count))) << task_index << "] " << setfill(' ');
#if (DEBUG_LEVEL > 1)
        printMatrix(&hash_state[0]);
#endif
        cout << setw(9) << setprecision(2) << fixed << local_duration.count() / 1000.0 << " ms. ";
        switch (can_solve)
        {
        case SOLVER_OK:
            cout << "Solved with depth " << path_length << " and " << nodes_alloc << " nodes allocated" << endl;
            break;
        case SOLVER_NO_SOLUTION:
            cout << "No solution" << endl;
            break;
        case SOLVER_CANNOT_SOLVE:
            cout << "Solution unreachable" << endl;
            break;
        }
#endif
        task_index += mpi_size;
    }

    auto pe_time = high_resolution_clock::now();
    auto pduration = duration_cast<microseconds>(pe_time - ps_time);
    double total_time = pduration.count() / 1000.0;

    MPI_Barrier(MPI_COMM_WORLD);

    double *runtimes = nullptr;
    if (mpi_rank == 0)
    {
        runtimes = new double[mpi_size];
    }
    MPI_Gather(&total_time, 1, MPI_DOUBLE, runtimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        double min_rtime = *min_element(runtimes, runtimes + mpi_size);
        double max_rtime = *max_element(runtimes, runtimes + mpi_size);

        cout << endl << "Workload balance:" << endl;
        for (int i = 0; i < mpi_size; ++i)
        {
            cout << "  Process " << setw(2) << i << " " << setprecision(2) << fixed << setw(10) << runtimes[i] << " ms" << endl;
        }
        delete[] runtimes;

        cout << endl << "  Min: " << setprecision(2) << fixed << min_rtime << ", Max: " << setprecision(2) << fixed << max_rtime << endl;
        cout << "  Balance: " << 100 - 100.0 * (max_rtime - min_rtime) / max_rtime << "%" << endl;
    }

    MPI_Win_free(&puzzle_win);
    if (mpi_rank == 0)
    {
        free(puzzles);
    }

    MPI_Finalize();
    return 0;
}

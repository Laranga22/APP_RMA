#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ERR_ARGS 1
#define ERR_IO   2
#define ERR_MEM  3
#define ERR_READ 4

int CUSTOM_UNIVERSE_SIZE;
#define CUSTOM_UNIVERSE_SIZE_VALUE 8
#define CRITICAL_SIZE (1000 * 1000)

void __rsv_set_comm_attr(MPI_Comm comm);

int main(int argc, char *argv[]) {
    int mpi_rank, mpi_size;
    int gsize[2];
    int lsize[2];
    int *mat = NULL;
    FILE *input_file;
    MPI_Comm parent_comm, child_comm;

    MPI_Init(&argc, &argv);
    __rsv_set_comm_attr(MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
        // This is the initial process
        if (argc < 4) {
            if (!mpi_rank) {
                printf("Usage: %s file rows cols\n\n", argv[0]);
            }
            MPI_Finalize();
            return ERR_ARGS;
        }

        gsize[0] = atoi(argv[2]);
        gsize[1] = atoi(argv[3]);

        // Determine the number of processes to spawn
        int optimal_processes = 1; // Default to 1 process
        if (gsize[0] * gsize[1] > CRITICAL_SIZE) {
            optimal_processes = CUSTOM_UNIVERSE_SIZE_VALUE;
        }

        char *argv_spawn[] = {argv[0], argv[1], argv[2], argv[3], NULL};
        int errcodes[optimal_processes];

        // Spawning child processes
        MPI_Comm_spawn(argv[0], argv_spawn, optimal_processes - 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &child_comm, errcodes);
        
        MPI_Bcast(gsize, 2, MPI_INT, 0, child_comm); // Broadcast in the spawned communicator

        parent_comm = child_comm;
    }

    MPI_Comm_rank(parent_comm, &mpi_rank);
    MPI_Comm_size(parent_comm, &mpi_size);

    MPI_Bcast(gsize, 2, MPI_INT, 0, parent_comm);

    lsize[0] = gsize[0] / mpi_size + (mpi_rank < gsize[0] % mpi_size ? 1 : 0);
    lsize[1] = gsize[1];

    int *shared_mat;
    MPI_Win win;
    MPI_Aint size = (mpi_rank == 0) ? gsize[0] * gsize[1] * sizeof(int) : 0;
    int disp_unit = sizeof(int);

    MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, parent_comm, &shared_mat, &win);

    int *send_counts = (int *)malloc(sizeof(int) * mpi_size);
    int *displs = (int *)malloc(sizeof(int) * mpi_size);

    if (mpi_rank == 0) {
        if (!(input_file = fopen(argv[1], "r"))) {
            fprintf(stderr, "Error opening input file or dimensions are wrong\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_IO);
        }
        if (!(mat = (int *)malloc(sizeof(int) * gsize[0] * gsize[1]))) {
            fprintf(stderr, "Error allocating memory\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
        }
        if (fread(mat, sizeof(int), gsize[0] * gsize[1], input_file) != gsize[0] * gsize[1]) {
            fprintf(stderr, "Error reading input file or dimensions are wrong\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_READ);
        }
        memcpy(shared_mat, mat, gsize[0] * gsize[1] * sizeof(int));
        free(mat);
        fclose(input_file);
    }

    MPI_Win_fence(0, win);

    if (mpi_rank != 0) {
        shared_mat = (int *)malloc(sizeof(int) * lsize[0] * lsize[1]);
    }

    displs[0] = 0;
    send_counts[0] = lsize[0] * lsize[1];
    for (int i = 1; i < mpi_size; i++) {
        displs[i] = displs[i - 1] + send_counts[i - 1];
        send_counts[i] = (gsize[0] / mpi_size + (i < gsize[0] % mpi_size ? 1 : 0)) * lsize[1];
    }

    MPI_Scatterv(shared_mat, send_counts, displs, MPI_INT,
                 mpi_rank ? shared_mat : MPI_IN_PLACE, lsize[0] * lsize[1], MPI_INT, 0, parent_comm);

    /* sum */
    int sum = 0;
    for (int i = 0; i < lsize[0]; ++i) {
        for (int j = 0; j < lsize[1]; ++j) {
            sum += shared_mat[i * lsize[1] + j];
        }
    }

    MPI_Reduce(mpi_rank ? &sum : MPI_IN_PLACE, &sum, 1, MPI_INT, MPI_SUM, 0, parent_comm);

    if (!mpi_rank) {
        printf("Sum is %d\n", sum);
    }

    MPI_Win_free(&win);

    MPI_Finalize();

    return 0; 
}

void __rsv_set_comm_attr(MPI_Comm comm) {
    int usk = 0;
    int usv = CUSTOM_UNIVERSE_SIZE_VALUE;

    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN,
                           &CUSTOM_UNIVERSE_SIZE, (void *)0);
    MPI_Comm_set_attr(comm, CUSTOM_UNIVERSE_SIZE, &usv);
}
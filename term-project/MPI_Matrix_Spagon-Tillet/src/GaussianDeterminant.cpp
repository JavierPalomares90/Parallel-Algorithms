/********************************
 * only works on square matrices of size n x n
 *******************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "mpi.h"
#include <time.h>


void readMatrix(FILE *fp, double *matrix, int dim);
void printMatrix(double *M, int dim);

double determinant(double *M, double *L, int n);
double determinant_mpi(double *M, double *L, int n, int world_size, int rank, int start, int num);

void copyMatrix(double *A, double *B, int n);


int main(int argc, char **argv) {
    char file[] = "matrix.txt";
    
    double *matrix;

    
    MPI_Init(&argc, &argv);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int dim;
    double det_seq, det_mpi;
    double result, det_block;
    double seq_start, seq_end, seq_elapsed;
    double mpi_start, mpi_end, mpi_elapsed;
    double mpi_maxelapsed;
    
    double *mbuf;
    
    FILE *fp;
    
    if (world_rank == 0) {
        printf("world size is %d\n", world_size);
        
        fp = fopen(file, "r");
        if (fp == NULL) {
            perror ("Error opening file");
            return -1;
        }
        else {
            char line[10000];
            char *s;
            if (fgets(line, sizeof line, fp) != NULL) {
                s = strtok(line, " ");
                if (s != NULL) dim = atoi(s);
                else {
                    perror ("Invalid matrix");
                    return -1;
                }
                if (dim < 2) {
                    perror ("Invalid matrix dimensions");
                    return -1;
                }
            }
            printf ("dim is %d\n", dim);
            

        }
    }
    matrix = (double *)std::malloc(dim * dim * sizeof(double));

    
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    double *L = (double *)std::malloc(dim * dim * sizeof(double));
    
    if (world_rank == 0) {
        readMatrix(fp, matrix, dim);
        fclose(fp);
        
        double *M = (double *)std::malloc(dim * dim * sizeof(double));
        copyMatrix(matrix, M, dim);

        seq_start = MPI_Wtime();
        det_seq = determinant(M, L, dim);
        seq_end = MPI_Wtime();
        seq_elapsed = seq_end - seq_start;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Bcast(matrix, dim * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    int small, big, numBig, start, num;
    small = dim / world_size;
    if (dim % world_size == 0) big = small;
    else big = small + 1;
    numBig = dim - (small * world_size);
    
    if (world_rank < numBig) {
        num = big;
        start = world_rank * big;
    } else {
        num = small;
        start = (world_rank * small) + numBig;
    }
    
    mpi_start = MPI_Wtime();
    
    result = determinant_mpi(matrix, L, dim, world_size, world_rank, start, num);
    
    
    
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    double inter;
    MPI_Reduce(&result, &inter, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    det_mpi = inter;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    mpi_end = MPI_Wtime();
    mpi_elapsed = mpi_end - mpi_start;
    
    MPI_Reduce(&mpi_elapsed, &mpi_maxelapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    
    if (world_rank == 0) {
        if (det_seq == det_mpi) {
            printf ("The determinants match! It is %f\n", det_mpi);
        } else {
            printf ("The determinants don't match :(\n");
            printf ("The sequential determinant is %f\n", det_seq);
            printf ("The mpi determinant is %f\n", det_mpi);
        }
        
        printf ("sequential time elapsed: %f\n", seq_elapsed);
        printf ("mpi time elapsed: %f\n", mpi_maxelapsed);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    
    return 0;
    
}

void readMatrix(FILE *fp, double *matrix, int dim) {
    char line[10000];
    char *s;
    for (int i=0; i<dim; i++) {
        if (fgets(line, sizeof line, fp) != NULL) {
            for (int j=0; j<dim; j++) {
                if (j==0) s = strtok(line, " ");
                else s = strtok(NULL, " ");
                if (s != NULL) {
                    int index = i * dim + j;
                    matrix[index] = atoi(s) / 10.0;
                } else perror ("Invalid Matrix");
            }
        } else {
            printf ("line read error\n");
        }
    }
}

void printMatrix(double *M, int dim) {
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            int index = i * dim + j;
            printf ("%f ", M[index]);
        }
        printf ("\n");
    }
}

double determinant(double *M, double *L, int n) {
    int i, j, k;
    double det;
    
    
    for (k = 0; k < n; k++) {
        for (i = k + 1; i < n; i++) {
            L[i * n + k] = M[i * n + k] / M[k * n + k];
        }
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                M[i * n + j] -= M[k * n + j] * L[i * n + k];
            }
            M[i * n + k] = 0.0;
        }
    }
    
    det = 1.0;
    for (i = 0; i < n; i++) {
        det = det * M[i * n + i];
    }
    return det;
}

double determinant_mpi(double *M, int n, int world_size, int rank, int start, int num) {
    int i, j, k;
    double tmp[n];
    double scaling;

    int cnt = 0;
    for (i=0; i<n; i++) {
        if (i % world_size == rank) { // if the reference row is assigned to me
            MPI_Bcast(M + (i * n), n, MPI_DOUBLE, rank, MPI_COMM_WORLD);
            for (j=0; j<n; j++) {
                tmp[j] = M[i * n + j];
            }
            
            cnt++;

        } else{
            MPI_Bcast(tmp, n, MPI_DOUBLE, i%world_size, MPI_COMM_WORLD); // receive updated ref row
            cnt++;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        for(j=cnt; j<n; j++){ // update all rows below the ref row
            scaling = M[j * n + i]/tmp[i];
            for(k=i; k<n; k++) {  // apply scaling to all entries in current row
                M[j * n + k] = M[j * n + k] - scaling*tmp[k];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
    double det = 1.0;
    for (i=start; i<start + num; i++) {
        det *= M[i * n + i];
    }
    return det;
}

                                
void copyMatrix(double *A, double *B, int n) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            B[i*n+j] = A[i*n+j];
        }
    }
}

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

void readMatrix(FILE *fp, double *matrix, int dim);
void printMatrix(double *M, int dim);

void reduceMatrix(double *M, double *R, int dim, int myCol);
double determinant(double *M, int dim);

int main(int argc, char **argv) {
    char file[] = "matrix.txt";
    
    MPI_Init(&argc, &argv);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    double *matrix;
    int dim;
    double det_seq, det_mpi, det_block, result;
    
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
    
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    matrix = (double *)std::malloc(dim * dim * sizeof(double));

    if (world_rank == 0) {
        readMatrix(fp, matrix, dim);
        fclose(fp);
        
        det_seq = determinant(matrix, dim);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Bcast(matrix, dim * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (dim == 2) {
        if (world_rank == 0) det_mpi = determinant(matrix, dim);
    } else {
        
        // implement recursive parallel determinant
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

        int newDim = dim - 1;
        result = 0.0;
        mbuf = (double *)std::malloc(newDim * newDim * sizeof(double));
        for (int i=start; i<start+num; i++) {
            reduceMatrix(matrix, mbuf, dim, i);
            double mid = matrix[i] * determinant(mbuf, newDim);
            if (i % 2 == 0) {
                result += mid;
            } else {
                result -= mid;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&result, &det_mpi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    
    if (world_rank == 0) {
        if (det_seq == det_mpi) {
            printf ("The determinants match! It is %f\n", det_mpi);
        } else {
            printf ("The determinants don't match :(\n");
            printf ("The sequential determinant is %f\n", det_seq);
            printf ("The mpi determinant is %f\n", det_mpi);
        }
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

void reduceMatrix(double *M, double *R, int dim, int myCol) {
    int row = 0;
    int col = 0;
    int rIndex, mIndex;
    int newDim = dim - 1;
    for (int i=1; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            if (j != myCol) {
                rIndex = row * newDim + col;
                mIndex = i * dim + j;
                R[rIndex] = M[mIndex];
                col++;
            }
        }
        row++;
        col = 0;
    }
}


double determinant(double *M, int dim) {
    double result = 0;
    if (dim == 2) {
        result = (M[0] * M[3]) - (M[1] * M[2]);
    } else {
        int newDim = dim - 1;
        double *mbuf = (double *)std::malloc(newDim * newDim * sizeof(double));
        for (int i=0; i<dim; i++) {
            reduceMatrix(M, mbuf, dim, i);
            double mid = (M[i] * determinant(mbuf, newDim));
            if (i % 2 == 0) {
                result += mid;
            } else {
                result -= mid;
            }
        }
        free(mbuf);
    }
    return result;
}

//Solving Linear Systems of Equations using Gaussian Elimination

//References
	//Books
		//Parallel Scientific Computing in C++ and MPI, (Karniadakis, and Kirby) - Chapter 9, Fast Linear Solvers
		//Introduction to Parallel Computing, (Grama, Gupta, Karypis, and Kumar) - Chapter 8, Dense Matrix Algorithms
		//Parallel Programming in C with MPI and OpenMP (Quinn) - Chapter 12, Solving Linear Systems
	//Course Notes
		//University of Nizhni Novgorod (Victor)

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <mpi.h>
#include <time.h>
#include <math.h>

using namespace std;

//GLOBAL VARIABLES
int numTasks = 0;	// Total number of processes
int taskID = 0;		// Process rank number

int *pivotPosArray = NULL;		//keeping track of the pivot
int *localPivotIter = NULL;		//local pivot iter

int *localInd = NULL;	// Number of the first row located on the processes
int *localNum = NULL;	// Number of the linear system rows located on the processes

/* Print a matrix of a certain size */
void PrintMatrix(double* matrix, int rowSize, int colSize) {
	for (int i = 0; i < rowSize; i++) {

		for (int j = 0; j < colSize; j++) {
			printf("%7.3f ", matrix[i*colSize + j]);
		}

		printf("\n");
	}
}

/* Print a vector of a certain length */
void PrintVector(double* vector, int length) {
	for (int i = 0; i < length; i++) {
		printf("%7.3f ", vector[i]);
	}

	printf("\n");
}

/* Initialize matrix to random values 
**
** Make it lower triangulare to ensure that it is solveable.
*/
void RandomDataInit(double* matrix, double* vector, int size) {
	//SEED VALUE
	srand(unsigned(clock()));

	//Initalize the vector
	for (int i = 0; i < size; i++) {
		vector[i] = rand() / double(1000);
	}

	//Initialize matrix to lower triangle to esnure that it is solveable
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (j <= i) {
				matrix[i*size + j] = rand() / double(1000);
			}
			else {
				matrix[i*size + j] = 0;
			}
		}

	}
}

/* SCATTER THE ROWS OF THE MATRIX TO EACH PROCESS */
void ScatterRows(double* matrix, double* locaRows, double* vector, double* localVector, int size, int rowNum) {
	int *sendCount; // sendCount USED TO SCATTERV
	int *sendInd; // INDEX ARRAY USED TO SCATTERV
	int rowsLeft = size; // NUMBER OF ROWS LEFT TO DISTRIBUTE

	//USED TO SCATTERV
	sendInd = new int[numTasks];
	sendCount = new int[numTasks];

	rowNum = (size / numTasks);
	sendCount[0] = rowNum*size;
	sendInd[0] = 0;

	for (int i = 1; i < numTasks; i++) {
		rowsLeft -= rowNum;
		rowNum = rowsLeft / (numTasks - i);
		sendCount[i] = rowNum*size;
		sendInd[i] = sendInd[i - 1] + sendCount[i - 1];
	}

	//SCATTER ROWS
	MPI_Scatterv(matrix, sendCount, sendInd, MPI_DOUBLE, locaRows, sendCount[taskID], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	rowsLeft = size;
	localInd[0] = 0;
	localNum[0] = size / numTasks;

	for (int i = 1; i < numTasks; i++) {
		rowsLeft -= localNum[i - 1];
		localNum[i] = rowsLeft / (numTasks - i);
		localInd[i] = localInd[i - 1] + localNum[i - 1];
	}

	//SCATTER VECTOR
	MPI_Scatterv(vector, localNum, localInd, MPI_DOUBLE, localVector, localNum[taskID], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[] sendInd;
	delete[] sendCount;
}

/* Column elimination during elimination */
void columnElimination(double* locaRows, double* localVector, double* pivotRow, int size, int rowNum, int iter) {
	double multiplier;

	for (int i = 0; i < rowNum; i++) {
		if (localPivotIter[i] == -1) {
			multiplier = locaRows[i*size + iter] / pivotRow[iter];
			for (int j = iter; j < size; j++) {
				locaRows[i*size + j] -= pivotRow[j] * multiplier;
			}

			localVector[i] -= pivotRow[size] * multiplier;
		}
	}
}

/* Perform elimination */
void GaussianElimination(double* locaRows, double* localVector, int size, int rowNum) {
	double maxValue;	// Value of the pivot element of thе process
	int pivotPos;		// Position of the pivot row in the process stripe

	// Structure used for ALLREDUCE, MPI_DOUBLE_INT, MAXLOC
	struct {
			double	maxValue; 
			int		taskID;
	} localPivot, pivot;

	double *pivotRow = new double[size + 1];

	for (int i = 0; i < size; i++) {
		// Calculating the local pivot row
		double maxValue = 0;

		for (int j = 0; j < rowNum; j++) {
			if ((localPivotIter[j] == -1) && (maxValue < fabs(locaRows[j*size + i]))) {
				maxValue = fabs(locaRows[j*size + i]);
				pivotPos = j;
			}
		}

		localPivot.maxValue = maxValue;
		localPivot.taskID = taskID;

		// Who has the max value? Find the global pivot amongst local pivots!
		MPI_Allreduce(&localPivot, &pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

		// Broadcasting the global pivot row
		if (taskID == pivot.taskID) {
			localPivotIter[pivotPos] = i;
			pivotPosArray[i] = localInd[taskID] + pivotPos;
		}

		MPI_Bcast(&pivotPosArray[i], 1, MPI_INT, pivot.taskID, MPI_COMM_WORLD);

		if (taskID == pivot.taskID) {
			for (int j = 0; j < size; j++) {
				pivotRow[j] = locaRows[pivotPos*size + j];
			}
			pivotRow[size] = localVector[pivotPos];
		}

		//Broadcast pivot row and eliminate
		MPI_Bcast(pivotRow, size + 1, MPI_DOUBLE, pivot.taskID, MPI_COMM_WORLD);
		columnElimination(locaRows, localVector, pivotRow, size, rowNum, i);

	}
}

/* Perform Back Substitution */
void BackSubstitution(double* locaRows, double* localVector, double* localResult, int size, int rowNum) {
	int iterProcRank;	// Rank of the process with the current pivot row
	int iterPivotPos;	// Position of the pivot row of the process
	double iterResult;	// Value of the current unknown
	double val;

	for (int i = size - 1; i >= 0; i--) {
		//FIND WHICH PROCESS HOLDS THE pivot ROW
		int rowIndex = pivotPosArray[i];

		for (int i = 0; i < numTasks - 1; i++) {
			if ((localInd[i] <= rowIndex) && (rowIndex < localInd[i + 1])) {
				iterProcRank = i;
			}
		}

		if (rowIndex >= localInd[numTasks - 1]) {
			iterProcRank = numTasks - 1;
		}

		iterPivotPos = rowIndex - localInd[iterProcRank];

		//CALCULATE UNKNOWN
		if (taskID == iterProcRank) {
			iterResult = localVector[iterPivotPos] / locaRows[iterPivotPos*size + i];
			localResult[iterPivotPos] = iterResult;
		}

		//BROADCAST UNKNOWN
		MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);

		//UPDATE VECTOR B
		for (int j = 0; j < rowNum; j++) {
			if (localPivotIter[j] < i) {
				val = locaRows[j*size + i] * iterResult;
				localVector[j] = localVector[j] - val;
			}
		}

	}
}

//////// MAIN

void main(int argc, char* argv[]) {
	// problem size (CHANGE THIS HERE)
	int size = 20;

	//PROBLEM DATA STRUCTURES (matrix stored as 1D)
	double *matrix = NULL, *vector = NULL, *result = NULL;

	//"LOCAL" to a process (used after scatter).
	double *locaRows = NULL, *localVector = NULL, *localResult = NULL;

	//NUMBER OF MATRIX ROWS
	int rowNum = 0;

	//TIMING ANALYSIS
	double start = 0, finish = 0, duration = 0;	

	//ROWS LEFT TO DISTRIBUTE
	int rowsLeft = 0;

	//INIT
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskID);
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);

	//DISPLAY CORE AMOUNT AND PROBLEM SIZE
	if (taskID == 0) {

		printf("\nnumber of cores = %d / size of matrix = %d\n\n", numTasks, size);

		if (numTasks > size) {
			printf("problem size must be larger than core amount.  (mustn't have idle cores!) \n");
			exit(-1);
		}

	}

	//LOCAL INIT
	rowsLeft = size;		// Number of rows that haven't been distributed yet

	for (int i = 0; i < taskID; i++)
		rowsLeft = rowsLeft - rowsLeft / (numTasks - i);

	rowNum = rowsLeft / (numTasks - taskID);
	locaRows = new double[rowNum*size];
	localVector = new double[rowNum];
	localResult = new double[rowNum];
	pivotPosArray = new int[size];
	localPivotIter = new int[rowNum];
	localInd = new int[numTasks];
	localNum = new int[numTasks];

	for (int i = 0; i < rowNum; i++)
		localPivotIter[i] = -1;			//initialize to -1

	if (taskID == 0) {
		matrix = new double[size*size];
		vector = new double[size];
		result = new double[size];
		RandomDataInit(matrix, vector, size);
	}



	//START TIMER
	start = MPI_Wtime();

	//SCATTER DATA
	ScatterRows(matrix, locaRows, vector, localVector, size, rowNum);

	//GAUSS ELIMINATION
	GaussianElimination(locaRows, localVector, size, rowNum);

	//BACK SUBSTITUTION
	BackSubstitution(locaRows, localVector, localResult, size, rowNum);

	//GATHER RESULTS
	MPI_Gatherv(localResult, localNum[taskID], MPI_DOUBLE, result, localNum, localInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//STOP TIMER
	finish = MPI_Wtime();
	duration = finish - start;




	//if (taskID == 0) {
	//	// Printing the result vector
	//	printf("\n Result Vector: \n");
	//	PrintVector(result, size);
	//}

	// DISPLAY RUNNING TIME
	if (taskID == 0)
		printf("\n Time of execution: %f\n", duration);



	//FINALIZE (CLEAN UP)
	if (taskID == 0) {
		delete[] vector;
		delete[] matrix;
		delete[] result;
	}

	delete[] locaRows;
	delete[] localVector;
	delete[] localResult;
	delete[] pivotPosArray;
	delete[] localPivotIter;
	delete[] localInd;
	delete[] localNum;

	MPI_Finalize();
}
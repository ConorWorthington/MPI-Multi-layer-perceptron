#include <iostream>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include "mpi.h"
//Global variable set up
const int radius = 3;
const int numSamples = 100;
const double learningRate = 0.15;
const int epochs = 1;
const int numNeurons = 2;
double start_time;
double end_time;
//Set up neurons
double inputLayer[2][numNeurons] = { 0 }; //takes input and weights
double outputLayer[1][numNeurons] = { 0 }; //takes weights and outputs
MPI_Status status;
MPI_Request request;

//Calculates dot product of two arrays from a given pointer and returns a total - must be same size
double dotProduct(double *array1, double *array2, int size) {
	double total = 0;
	for (int i = 0; i < size; i++) {
		total += array1[i] * array2[i];
	}
	return total;
}

//Does an element by element multiplication but keeps the size of the array the same- alters an array at a given pointer to contain this
void elementMultiply(double *array1, double *array2, double *output, int rows) {
	for (int i = 0; i < rows; i++) {
		output[i] = array1[i] * array2[i];
	}
}
//Calculates the sigmoid derivative for every position in an array then alters an array at a pointer to contain this 
void sigmoidDerivativeMatrix(double* inputArray, int rows, double *output) {
	for (int i = 0; i < rows; i++) {
		output[i] = inputArray[i] * (1 - inputArray[i]);
	}
}
//Calculates the sigmoid derivative of a singular value and returns it as a double
double sigmoidDerivativeScalar(double inputVal) {
	double sigmoidValue;
	sigmoidValue = inputVal * (1 - inputVal);
	return sigmoidValue;
}

//Calculates the sigmoid value of a single input and returns it as a double
double sigmoidScalar(double inputVal) {
	double sigmoidValue;
	sigmoidValue = 1 / (1 + exp(-inputVal));
	return sigmoidValue;
}
//Calculates the sigmoid for every position in an array then alters an array at a pointer to contain this 
void sigmoidMatrix(double *inputArray, int rows, double *output) {
	for (int i = 0; i < rows; i++) {
		output[i] = 1 / (1 + exp(-inputArray[i]));
	}
}
//Completes the matrix multiplication of two arrays and puts the values in a given pointer output
void mpiMatrixMultiply(double *array1, double *array2, double *output, int arr1_rows, int arr1_cols, int arr2_cols) {
	int rank, world_size, strip, workers, start, end, iii;
	double result;
	double * tmpResult;
	tmpResult = (double*)malloc(arr1_rows*arr2_cols * sizeof(double));
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	workers = world_size - 1;
	start = 0;
	iii = 0;
	end = arr1_rows;
	if (rank == 0) {//set up master  rank
		start_time = MPI_Wtime();
		strip = arr1_rows / workers; //minimum number of rows per process *ideally
		for (int reciever = 1; reciever <= workers; reciever++) { //For all ranks bar the master rank 0
			start = (reciever - 1)*strip;
			if ((((reciever + 1) == world_size) && ((arr1_rows % (world_size - 1)) != 0))) {
				end = arr1_rows;//Allocate remaining rows
			}
			else {
				end = start + strip;//Set end point to a given point on array
			}//Send out information for other ranks to complete multiplication 
			MPI_Send(&start, 1, MPI_INT, reciever, 1, MPI_COMM_WORLD);
			MPI_Send(&end, 1, MPI_INT, reciever, 1, MPI_COMM_WORLD);
			MPI_Send(array1, arr1_rows*arr1_cols, MPI_DOUBLE, reciever, 1, MPI_COMM_WORLD);//Update arrays incase values different due to only rank 0 getting updated matrix multi values
			MPI_Send(array2, arr1_cols*arr2_cols, MPI_DOUBLE, reciever, 1, MPI_COMM_WORLD);

		}

		for (int i = 1; i <= workers; i++) { //Recieve from every rank in order 
			MPI_Recv(&start, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//Recieve start position 
			MPI_Recv(&end, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//End position
			MPI_Recv(&iii, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//Number of items
			MPI_Recv(&output[start*iii],iii, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);//Update array


		}
		end_time = MPI_Wtime();

	}
	if (rank > 0) {
		//Update workers
		MPI_Recv(&start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(array1, arr1_rows*arr1_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(array2, arr1_cols*arr2_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Send(&start, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);//Send off new start
		MPI_Send(&end, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);//Send off new end
		for (int row = start; row < end; row++) {
			for (int col = 0; col < arr2_cols; col++) {
				result = 0.0;
				for (int i = 0; i < arr1_cols; i++)
				{
					result = result + array1[row * arr1_cols + i] * array2[i * arr2_cols + col];//Complete matrix multi between start and end
				}
				tmpResult[iii] = result;
				iii = iii + 1;//Update counter so that size is known

			}
		}
		MPI_Send(&iii, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);//Send size
		MPI_Send(tmpResult, iii, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);//Send array


	}
	free(tmpResult);//Clear temporary array to avoid memory leak

}
//Adds two matricies together and returns in pointer of the first matrix
void addMatrix(double *targetArray, double *addArray, int rows) {
	for (int i = 0; i < rows; i++) {
		targetArray[i] = targetArray[i] + addArray[i];
	}
}
//Creates input data and returns a pointer to the data set 
double* createInput(int dataSize, double* inputData) {//Takes in data to ensure inputs match outputs
	int setSize = dataSize * 2;
	double* dataSet = new double[setSize];
	for (int i = 0; i < dataSize; i++) {
		dataSet[i] = radius * cos(inputData[i]); //Uses trig to build data set 
		dataSet[(dataSize + i)] = radius * sin(inputData[i]);//As dataset has two co-ords was more efficient to return in one array of double size
	}
	return dataSet;
}
//Creates an output of random angles between 0 and 1 with 4 decimal places
double* createOutput(int dataSize) {
	double* dataSet = new double[dataSize];
	for (int i = 0; i < dataSize; i++) {
		double theta = rand() % 1000; //randomly seeded number generator
		theta = theta / 1000.0;
		dataSet[i] = theta;
	}
	return dataSet;
}
//Does the training and error calculation for the Multi layer perceptron - taking in inputs and expected outputs to perform supervised learning 
void trainMultiLayerPerceptron(double* inputData, double* expectedOutputData, int maxiter) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//Initialises rank 
	for (int j = 0; j < maxiter; j++) {
		double layerOneAdjustment[2][numNeurons] = { 0 }; //Initalise the adjustment arrays to nothing
		double layerTwoAdjustment[1][numNeurons] = { 0 };
		double errorSum = 0.0;//reset error to 0
		for (int i = 0; i < numSamples; i++) {
			//Initialise arrays which hold temporary variables and returns from matrix multiplications to 0.0
			double layer1output[1][numNeurons] = { 0 };
			double inputDataArray[1][2] = { inputData[i] ,inputData[numSamples + i] };
			double transposeInputData[2][1] = { inputData[i],inputData[numSamples + i] };
			double layerOneAdjustmentTmp[2][numNeurons] = { 0 };
			double layerTwoAdjustmentTmp[1][numNeurons] = { 0 };
			double layer2delta[1][1] = { 0 };
			double layer1error[1][numNeurons] = { 0 };
			double layer1delta[1][numNeurons] = { 0 };
			double layer1outputSigmoid[1][numNeurons] = { 0 };
			mpiMatrixMultiply(*inputDataArray, *inputLayer, *layer1output, 1, 2, numNeurons);//Calculate layer 1 output
			sigmoidMatrix(*layer1output, numNeurons, *layer1output);//Sigmoid of output
			double layer2output = sigmoidScalar(dotProduct(*layer1output, *outputLayer, numNeurons));//Calculate sigmoid of dot product for layer 2 output
			double layer2error = expectedOutputData[i] - layer2output;//Calculate layer 2 error
			layer2delta[0][0] = layer2error * sigmoidDerivativeScalar(layer2output);//Calculate layer 2 delta into an array 
			mpiMatrixMultiply(*layer2delta, *outputLayer, *layer1error, 1, 1, numNeurons);//Calculate layer 1 error using matrix multi of level2delta and outputlayer
			sigmoidDerivativeMatrix(*layer1output, numNeurons, *layer1outputSigmoid); //Calculate sigmoid derivative of layer 1 output
			elementMultiply(*layer1error, *layer1outputSigmoid, *layer1delta, numNeurons); //Element by element multiply layer 1 error by sigmoid
			mpiMatrixMultiply(*transposeInputData, *layer1delta, *layerOneAdjustmentTmp, 2, 1, numNeurons);//Calculate layer one adjustment into a temporary array
			mpiMatrixMultiply(*layer2delta, *layer1output, *layerTwoAdjustmentTmp, 1, 1, numNeurons);//Calculate layer two adjustment into a temporary array
			for (int ii = 0; ii < numNeurons; ii++) {//Iterate through and update layer adjustments using temporary array
				layerOneAdjustment[0][ii] = layerOneAdjustment[0][ii] + layerOneAdjustmentTmp[0][ii];
				layerOneAdjustment[1][ii] = layerOneAdjustment[1][ii] + layerOneAdjustmentTmp[1][ii];
				layerTwoAdjustment[0][ii] = layerOneAdjustment[0][ii] + layerTwoAdjustmentTmp[0][ii];
			}			
			errorSum = errorSum + (layer2error * layer2error);//Calculate error sum
			if (j == (maxiter - 1) && rank == 0) {
				printf("input 1 %lf\n", inputData[i]);
				printf("input 2 %lf\n", inputData[numSamples + i]);
				printf("Expected output %lf\n", expectedOutputData[i]);
				printf("output 1 %lf\n", layer2output);
				printf("error sum %lf\n", errorSum);
			}
			for (int ii = 0; ii < numNeurons; ii++) {//Update layer weights by learning rate times the adjustment 
				inputLayer[0][ii] = inputLayer[0][ii] + learningRate * layerOneAdjustment[0][ii];
				inputLayer[1][ii] = inputLayer[1][ii] + learningRate * layerOneAdjustment[1][ii];
				outputLayer[0][ii] = outputLayer[0][ii] + learningRate * layerTwoAdjustment[0][ii];
			}

		}

	}
}
int main(int argc, char **argv)
{
	srand(time(NULL));//Seed random
	MPI_Init(NULL, NULL);//Set up MPI
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double* expectedOutputData = createOutput(numSamples);//Generate expected results
	double* inputData = createInput(numSamples, expectedOutputData); //Generate inputs
		for (int i = 0; i < epochs; i++) {
			for (int ii = 0; ii < numNeurons; ii++) {//Loop to setup Neural network weights to random values with bias -0.5
				inputLayer[0][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
				inputLayer[1][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
				outputLayer[0][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
			}
			trainMultiLayerPerceptron(inputData, expectedOutputData, 1000);//train network
		}
	printf("\nRunning Time = %f rank %d\n\n", end_time - start_time,rank);
	MPI_Finalize(); //end MPI
	return 0;
}


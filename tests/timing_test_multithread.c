#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "csv.h"
#include "lut.h"
#include "matrix_math.h"
#include "stft.h"

#include "functions.h"

#define TEST_ITERATIONS_DEF		(4096*128)

typedef struct thread_times_st {
	double total_time_us;
} thread_times_t;

int cloneMatrix(matrix32f_t *src, matrix32f_t *dst);
int cloneMatrix_c(matrix32c_t *src, matrix32c_t *dst);
void *threadFunction(void* times_arg);

// Global Variables
pthread_barrier_t init_barrier;
size_t test_iterations;

// Input/Output Matrices; read by all threads
size_t w1, h1, w2=0, h2=0, wo, ho;
// Matrices to store input and output data; Threads will copy the values found here
matrix32f_t input1, input2;
matrix32c_t cinput1, cinput2;
function_t selected_function = 0;

// LUT Pointers
lut32f_t *lut0_ptr, *lut1_ptr;

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Matrix Routines Multithreaded Timing Tests (%d functions)", valid_function_count);
#ifndef SERIAL
	printf(" (NEON)");
#endif
#ifdef DEBUG
	printf(" [Debug Build]");
#endif
	printf("\n\n");

	if(argc!=8 && argc!=5) {
		printf("Usage: timing_test [function] [h1] [w1] [input1] [h2] [w2] [input2]\nFunctions can be: ");
		for(uint8_t f = 0; f < valid_function_count; f++) { printf("%s ", valid_functions_str[f]); }
		printf("\n\n");
		return 1;
	}

	input1.d = NULL; input2.d = NULL;
	cinput1.d = NULL; cinput2.d = NULL;

	// Set up test iterations count; This might be reduced depending on the tested function
	test_iterations = TEST_ITERATIONS_DEF;

	// LUTs
	lut32f_t lut0, lut1;
	lut0.data = NULL;
	lut1.data = NULL;
	lut0_ptr = &lut0;
	lut1_ptr = &lut1;

	// Check function is valid
	selected_function = 0;
	while(strcmp(valid_functions_str[selected_function], argv[1])) {
		selected_function++;
		if(selected_function == valid_function_count) { break; }
	}
	if(selected_function == valid_function_count) {
		printf("Error: Function %s is invalid.\nPossible options are:\n\t", argv[1]);
		for(uint8_t i = 0; i < valid_function_count; i++) { printf("%s ", valid_functions_str[i]); }
		ret = 2; goto exit;
	}
	else { printf("\nSelected Function: %s (#%d)\n", valid_functions_str[selected_function], (uint32_t)selected_function); }

	// Get dimensions for file names
	h1 = atoi(argv[2]);	w1 = atoi(argv[3]);
	if(selected_function <= hadamardProductEnum || selected_function == hadamardProduct_complexEnum || selected_function == hadamardProduct_cbrEnum) { // Some functions require 2 inputs
		h2 = atoi(argv[5]); w2 = atoi(argv[6]);
	}

	// Configure input/output data dimensions and load LUTs
	switch(selected_function) {
		case matrixSumEnum:
			test_iterations *= 4;
			wo = w1; ho = h1; break;
		case matrixDiffEnum:
			test_iterations *= 4;
			wo = w1; ho = h1; break;
		case multVecByMatEnum:
			test_iterations /= 128;
			ho = 1; wo = w2; break;
		case multMatByVecEnum:
			test_iterations /= 128;
			ho = h1; wo = 1; break;
		case hadamardProductEnum:
			test_iterations *= 4;
			wo = w1; ho = h1; break;
		case elementwisePow2Enum:
			test_iterations *= 4;
			wo = w1; ho = h1; break;
		case reluEnum:
			test_iterations *= 4;
			wo = w1; ho = h1; break;
		case sqrtLutEnum:
			if(load32fLUT(&lut0, "lut/sqrt.lut")) {
				printf("Error: Could not load Square Root LUT.\n\n");
				ret = 10; goto exit;
			};
			wo = w1; ho = h1; break;
		case tanhLutEnum:
			if(load32fLUT(&lut0, "lut/tanh.lut")) {
				printf("Error: Could not load Tanh LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case sigmoidLutEnum:
			if(load32fLUT(&lut0, "lut/sigmoid.lut")) {
				printf("Error: Could not load Sigmoid LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case flipEnum:
			test_iterations *= 4;
			wo = w1; ho = h1; break;
		case extend2Enum:
			wo = w1*2; ho = h1; break;
		case extend4Enum:
			wo = w1*4; ho = h1; break;
		case extend8Enum:
			wo = w1*8; ho = h1; break;
		case squaredMagnitudeEnum:
			wo = w1; ho = h1; break;
		case hadamardProduct_complexEnum:
			wo = w1; ho = h1; break;
		case angleLutEnum:
			if(load32fLUT(&lut0, "lut/atan.lut")) {
				printf("Error: Could not load atan LUT (angle).\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case expiLutEnum:
			if(load32fLUT(&lut0, "lut/sin.lut")) {
				printf("Error: Could not load sine LUT.\n\n");
				ret = 10; goto exit;
			}
			if(load32fLUT(&lut1, "lut/cos.lut")) {
				printf("Error: Could not load cosine LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case hadamardProduct_cbrEnum:
			test_iterations /= 2048;
			wo = w1; ho = h1; break;
		default:
			ret = -2; goto exit;
	}

	// Load input(s) and configure output
	uint8_t test = 0;

	// Expect 2 inputs
	if(selected_function <= hadamardProductEnum) {
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK!\n");

		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h2, w2, &input2)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[7]);
			ret = 3; goto exit;
		}
		printf("OK!\n");
	}
	// Expect 1 (real) input
	else if(selected_function <= extend8Enum){
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK!\n");
	}
	// Expect 1 complex input
	else if(selected_function == squaredMagnitudeEnum){
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK!\n");

	}
	else if(selected_function == hadamardProduct_complexEnum){ // Expect 2 complex input
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK!\n");

		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h2, w2*2, (matrix32f_t*)&cinput2)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[7]);
			ret = 3; goto exit;
		}
		cinput2.w /= 2;
		printf("OK!\n");
	}
	else if(selected_function == angleLutEnum) { // Expect complex input, config. real output
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK!\n");

	}
	else if(selected_function == expiLutEnum) { // Expect real input, config. complex output
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK!\n");
	}
	// Expect one real and one complex input, config. complex output
	else if(selected_function == hadamardProduct_cbrEnum) {
		// Complex input (first)
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK!\n");

		// Real input (second)
		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK!\n");
	}

	// Barrier passed when all threads are done initializing
	pthread_barrier_init(&init_barrier, NULL, THREADS+1);

	// Structs for counting time
	struct timespec start_time, end_time;
	clock_getres(CLOCK_MONOTONIC, &start_time);

	pthread_t thread[THREADS];
	thread_times_t times[THREADS];
	// Initialize threads
	for(int i = 0; i < THREADS; i++) {
		pthread_create(&thread[i], NULL, threadFunction, &times[i]);
	}
	// Wait for threads to initialize
	pthread_barrier_wait(&init_barrier);

	// Wait for threads to complete
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	for(int i = 0; i < THREADS; i++) { pthread_join(thread[i], NULL); }
	clock_gettime(CLOCK_MONOTONIC, &end_time);


	// Clock measures resource time, not actual time
	// (e.g. if two cores worked for 1 sec. each, 2 sec. are counted)
	double start_t = start_time.tv_sec + (start_time.tv_nsec * 1e-9);
	double end_t = end_time.tv_sec + (end_time.tv_nsec * 1e-9);
	double total_time_us = (end_t - start_t) * 1e+6;

	double mean_thread_time_us = 0.0;
	for(int i = 0; i < THREADS; i++) {
		mean_thread_time_us += times->total_time_us;
	}
	mean_thread_time_us /= (double)THREADS;

	printf("Done testing!\n\n");
	printf("\tMultithreading Results\n");
	printf("\t===================================================\n");
	printf("\t Time for %4dK iterations: %4.3f s\n", test_iterations/1000, total_time_us/1e6 );
	printf("\t Mean Time/thread.:\t%2.1f ms\n", mean_thread_time_us/1e3);
	printf("\t Mean Time/iter.:\t%2.1f us\n", mean_thread_time_us/(double)test_iterations);
	printf("\t===================================================\n\n");

exit:
	deleteLUT32f(&lut0);
	deleteLUT32f(&lut1);
	deleteMatrix(&input1);
	deleteMatrix(&input2);
	deleteMatrix((matrix32f_t*)&cinput1);
	deleteMatrix((matrix32f_t*)&cinput2);
	return ret;
}


void* threadFunction(void* times_arg) {
	// Thread-level matrices
	matrix32f_t th_input1, th_input2, th_output1;
	matrix32c_t th_cinput1, th_cinput2, th_coutput1;
	th_input1.d = NULL; th_input2.d = NULL; th_output1.d = NULL;
	th_cinput1.d = NULL; th_cinput2.d = NULL; th_coutput1.d = NULL;

	// Clone inputs; Only inputs that have non-NULL data are cloned
	int test = 0;
	if(input1.d)	{ test |= cloneMatrix(&input1, &th_input1); }
	if(input2.d)	{ test |= cloneMatrix(&input2, &th_input2); }
	if(cinput1.d)	{ test |= cloneMatrix_c(&cinput1, &th_cinput1); }
	if(cinput2.d)	{ test |= cloneMatrix_c(&cinput2, &th_cinput2); }
	if(test) {
		// go to barrier now:(
		pthread_barrier_wait(&init_barrier);
		goto exit;
	}

	// Allocate output matrices depending on the selected_function
	// Expect 2 inputs
	if(selected_function <= hadamardProductEnum) {
		// Create matrix for our output
		if(newMatrix32f(ho, wo, &th_output1)) {
			goto exit;
		}
	}
	// Expect 1 (real) input
	else if(selected_function <= extend8Enum){
		// Create matrix for our output
		if(newMatrix32f(ho, wo, &th_output1)) {
			goto exit;
		}
	}
	else if(selected_function == squaredMagnitudeEnum){ // Create 1 real output
		// Create read matrix for our output
		if(newMatrix32f(ho, wo, &th_output1)) {
			goto exit;
		}
	}
	else if(selected_function == hadamardProduct_complexEnum){ // Expect 2 complex inputs
		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&th_coutput1)) {
			goto exit;
		}
		th_coutput1.w /= 2;
	}
	else if(selected_function == angleLutEnum) { // Expect complex input, config. real output
		// Create matrix for our output
		if(newMatrix32f(ho, wo, (matrix32f_t*)&th_output1)) {
			goto exit;
		}
	}
	else if(selected_function == expiLutEnum) { // Expect real input, config. complex output
		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&th_coutput1)) {
			goto exit;
		}
		th_coutput1.w /= 2;
	}
	// Expect one real and one complex input, config. complex output
	else if(selected_function == hadamardProduct_cbrEnum) {
		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&th_coutput1)) {
			goto exit;
		}
		th_coutput1.w /= 2;
	}

	// Ready to start
	struct timespec loop_start, loop_end;
	pthread_barrier_wait(&init_barrier);
	clock_gettime(CLOCK_REALTIME, &loop_start);
	for(size_t iter = 0; iter < test_iterations; iter++) {
		// printf("iter: %lu/%lu\n", iter, test_iterations);
		// Perform test
		switch(selected_function) {
			case matrixSumEnum:
				matrixSum(&th_input1, &th_input2, &th_output1); break;
			case matrixDiffEnum:
				matrixSum(&th_input1, &th_input2, &th_output1); break;
			case multVecByMatEnum:
				multVecByMat(&th_input1, &th_input2, &th_output1);	break;
			case multMatByVecEnum:
				multMatByVec(&th_input1, &th_input2, &th_output1); break;
			case hadamardProductEnum:
				hadamardProduct(&th_input1, &th_input2, &th_output1); break;
			case elementwisePow2Enum:
				elementwisePow2(&th_input1, &th_output1); break;
			case reluEnum:
				relu(&th_input1, &th_output1); break;
			case sqrtLutEnum:
				sqrtLUT(&th_input1, lut0_ptr, &th_output1); break;
			case tanhLutEnum:
				clampingLUT(&th_input1, lut0_ptr, &th_output1); break;
			case sigmoidLutEnum:
				clampingLUT(&th_input1, lut0_ptr, &th_output1); break;
			case flipEnum:
				flipVector(&th_input1, &th_output1); break;
			case extend2Enum:
				extendInput(&th_input1, &th_output1, 2); break;
			case extend4Enum:
				extendInput(&th_input1, &th_output1, 4); break;
			case extend8Enum:
				extendInput(&th_input1, &th_output1, 8); break;
			case squaredMagnitudeEnum:
				squaredMagnitude(&th_cinput1, &th_output1); break;
			// note: The following complex matrix math functions are hard-coded to have no output arg.
			case hadamardProduct_complexEnum:
				hadamardProduct_complex(&th_cinput1, &th_cinput2, &th_coutput1); break;
			case angleLutEnum:
				angleLUT_c(&th_cinput1, lut0_ptr, &th_output1); break;
			case expiLutEnum:
				expiLUT(&th_input1, lut0_ptr, lut1_ptr, &th_coutput1); break;
			case hadamardProduct_cbrEnum:
				hadamardProduct_cbr(&th_cinput1, &th_input1, &th_coutput1); break;
			default:
				goto exit;
		}
	}
	// Calculate time
	clock_gettime(CLOCK_REALTIME, &loop_end);
	thread_times_t* times = (thread_times_t*)times_arg;
	times->total_time_us = (((double)loop_end.tv_sec + (double)loop_end.tv_nsec*1e-9) - ((double)loop_start.tv_sec + (double)loop_start.tv_nsec*1e-9)) * 1e+6;


exit:
	// Clear data and just exit
	deleteMatrix(&th_input1); deleteMatrix(&th_input2);
	deleteMatrix(&th_output1);
	deleteMatrix((matrix32f_t*)&th_cinput1); deleteMatrix((matrix32f_t*)&th_cinput2);
	deleteMatrix((matrix32f_t*)&th_coutput1);
}

int cloneMatrix(matrix32f_t *src, matrix32f_t *dst) {
	if(newMatrix32f(src->h, src->w, dst))
		return 1;
	else
		return 0;
}

int cloneMatrix_c(matrix32c_t *src, matrix32c_t *dst) {
	if(newMatrix32f(src->h, src->w*2, (matrix32f_t*)dst))
		return 1;
	else {
		dst->w /= 2;
		return 0;
	}


}

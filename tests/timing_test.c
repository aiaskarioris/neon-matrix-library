#include <stdio.h>
#include <string.h>

#include "csv.h"
#include "clock.h"
#include "lut.h"
#include "matrix_math.h"
#include "stft.h"

#include "functions.h"

#ifdef SERIAL
	#define TEST_ITERATIONS		(256)
#else
	#define TEST_ITERATIONS		(4096*256)
#endif

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Matrix Routines Timing Tests (%d functions)", valid_function_count);
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

	// Struct to store input and output data
	matrix32f_t input1, input2, output1;
	matrix32c_t cinput1, cinput2, coutput1;
	input1.d = NULL; input2.d = NULL; output1.d = NULL;
	cinput1.d = NULL; cinput2.d = NULL; coutput1.d = NULL;

	// LUTs
	lut32f_t lut0, lut1;
	lut0.data = NULL;
	lut1.data = NULL;

	// Check function is valid
	function_t selected_function = 0;
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

	// Get dimensions from file names
	size_t w1, h1, w2=0, h2=0, wo, ho;
	h1 = atoi(argv[2]);	w1 = atoi(argv[3]);
	if(selected_function <= hadamardProductEnum || selected_function == hadamardProduct_complexEnum || selected_function == hadamardProduct_cbrEnum) { // Some functions require 2 inputs
		h2 = atoi(argv[5]); w2 = atoi(argv[6]);
	}

	// Configure input/output data dimensions and load LUTs
	switch(selected_function) {
		case matrixSumEnum:
			wo = w1; ho = h1; break;
		case matrixDiffEnum:
			wo = w1; ho = h1; break;
		case multVecByMatEnum:
			ho = 1; wo = w2; break;
		case multMatByVecEnum:
			ho = h1; wo = 1; break;
		case hadamardProductEnum:
			wo = w1; ho = h1; break;
		case elementwisePow2Enum:
			wo = w1; ho = h1; break;
		case reluEnum:
			wo = w1; ho = h1; break;
		case sqrtLutEnum:
			if(load32fLUT(&lut0, "lut/sqrt256.lut")) {
				printf("Error: Could not load Square Root LUT.\n\n");
				ret = 10; goto exit;
			};
			wo = w1; ho = h1; break;
		case tanhLutEnum:
			if(load32fLUT(&lut0, "lut/tanh65537.lut")) {
				printf("Error: Could not load Tanh LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case sigmoidLutEnum:
			if(load32fLUT(&lut0, "lut/sigmoid65537.lut")) {
				printf("Error: Could not load Sigmoid LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case flipEnum:
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
			if(load32fLUT(&lut0, "lut/atan65537.lut")) {
				printf("Error: Could not load atan LUT (angle).\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case expiLutEnum:
			if(load32fLUT(&lut0, "lut/sin256.lut")) {
				printf("Error: Could not load sine LUT.\n\n");
				ret = 10; goto exit;
			}
			if(load32fLUT(&lut1, "lut/cos256.lut")) {
				printf("Error: Could not load cosine LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case hadamardProduct_cbrEnum:
			wo = w1; ho = h1; break;
		default:
			ret = -2; goto exit;
	}

	// Load input(s) and configure output
	uint8_t test = 0;
	startClock();

	// Expect 2 inputs
	if(selected_function <= hadamardProductEnum) {
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		startClock();
		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h2, w2, &input2)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[7]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for our output
		if(newMatrix32f(ho, wo, &output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
	}
	// Expect 1 (real) input
	else if(selected_function <= extend8Enum){
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for our output
		if(newMatrix32f(ho, wo, &output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
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
	else if(selected_function == hadamardProduct_complexEnum){ // Expect 1 complex input
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&coutput1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
	}
	else if(selected_function == angleLutEnum) { // Expect complex input, config. real output
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for our output
		if(newMatrix32f(ho, wo, (matrix32f_t*)&output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
	}
	else if(selected_function == expiLutEnum) { // Expect real input, config. complex output
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&coutput1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
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
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Real input (second)
		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));


		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&coutput1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
	}

	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	uint32_t best_time_idx = -1, worst_time_idx = -1;
	clock_t start_time = clock();
	for(size_t iter = 0; iter < TEST_ITERATIONS; iter++) {
		// printf("\r%u/%u", iter, TEST_ITERATIONS);

		// Perform test
		switch(selected_function) {
			case matrixSumEnum:
				startClock(); matrixSum(&input1, &input2, &output1); break;
			case matrixDiffEnum:
				startClock(); matrixSum(&input1, &input2, &output1); break;
			case multVecByMatEnum:
				startClock(); multVecByMat(&input1, &input2, &output1);	break;
			case multMatByVecEnum:
				startClock(); multMatByVec(&input1, &input2, &output1); break;
			case hadamardProductEnum:
				startClock(); hadamardProduct(&input1, &input2, &output1); break;
			case elementwisePow2Enum:
				startClock(); elementwisePow2(&input1, &output1); break;
			case reluEnum:
				startClock(); relu(&input1, &output1); break;
			case sqrtLutEnum:
				startClock(); sqrtLUT(&input1, &lut0, &output1); break;
			case tanhLutEnum:
				startClock(); clampingLUT(&input1, &lut0, &output1); break;
			case sigmoidLutEnum:
				startClock(); clampingLUT(&input1, &lut0, &output1); break;
			case flipEnum:
				startClock(); flipVector(&input1, &output1); break;
			case extend2Enum:
				startClock(); extendInput(&input1, &output1, 2); break;
			case extend4Enum:
				startClock(); extendInput(&input1, &output1, 4); break;
			case extend8Enum:
				startClock(); extendInput(&input1, &output1, 8); break;
			// note: Complex matrix math functions are hard-coded to have no output arg.
			case squaredMagnitudeEnum:
				startClock(); squaredMagnitude(&cinput1, &output1); break;
			case hadamardProduct_complexEnum:
				startClock(); hadamardProduct_complex(&cinput1, &cinput2, &coutput1); break;
			case angleLutEnum:
				startClock(); angleLUT_c(&cinput1, &lut0, &output1); break;
			case expiLutEnum:
				startClock(); expiLUT(&input1, &lut0, &lut1, &coutput1); break;
			case hadamardProduct_cbrEnum:
				startClock(); hadamardProduct_cbr(&cinput1, &input1, &coutput1); break;
			default:
				ret = -2; goto exit;
		}

		// // Check timer: THIS IS REALLY SLOW !!!!!!!!!!!!
		// float last_time = readClock();
		// // Update best and worst time if needed
		// best_time_idx 	= (last_time < best_time)  ? iter : best_time_idx;
		// worst_time_idx 	= (last_time > worst_time) ? iter : worst_time_idx;
		// best_time  		= (last_time < best_time)  ? last_time : best_time;
		// worst_time 		= (last_time > worst_time) ? last_time : worst_time;
	}
	clock_t end_time = clock();
	double mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (double)TEST_ITERATIONS;

	printf("Done testing!\n\n");
	printf("\tResults\n");
	printf("\t===================================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", TEST_ITERATIONS, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);

	printf("\t===================================================\n\n");

exit:
	deleteLUT32f(&lut0);
	deleteLUT32f(&lut1);
	deleteMatrix(&input1);
	deleteMatrix(&input2);
	deleteMatrix(&output1);
	deleteMatrix((matrix32f_t*)&cinput1);
	deleteMatrix((matrix32f_t*)&cinput2);
	deleteMatrix((matrix32f_t*)&coutput1);
	return ret;
}

#include <stdio.h>
#include <string.h>

#include "csv.h"
#include "clock.h"
#include "matrix_math.h"

static const char* const	matrix_path[] = {
	"parameters/input_scale_bass.csv", "parameters/input_scale_drums.csv", "parameters/input_scale_vocals.csv", "parameters/input_scale_others.csv",
	"parameters/input_mean_bass.csv", "parameters/input_mean_drums.csv", "parameters/input_mean_vocals.csv", "parameters/input_mean_others.csv",
	"parameters/output_scale_bass.csv", "parameters/output_scale_drums.csv", "parameters/output_scale_vocals.csv", "parameters/output_scale_others.csv",
	"parameters/output_mean_bass.csv", "parameters/output_mean_drums.csv", "parameters/output_mean_vocals.csv", "parameters/output_mean_others.csv",
};
static const size_t 		matrix_dims[] = {/*input*/ 1, 1487, /*output*/ 1, 2049};
static const char* const 	input_path[] = { "csv/test1x1487.csv", "csv/test1x2049.csv" };

int main(int argc, char **argv) {
	uint8_t ret = 0, test = 0;
	printf("Aias Karioris, 2025\n");
	printf("Input/Output Shift-Scale Operations Timing Tests");
#ifndef SERIAL
	printf(" (NEON)");
#endif
#ifdef DEBUG
	printf(" [Debug Build]");
#endif
	printf("\n\n");

	if(argc > 2) {
		printf("Usage: %s [iterations]\n\n", argv[0]);
		return 1;
	}

	// Get number of iterations or default to 16
	uint32_t iterations = (argc==2) ? atoi(argv[1]) : 16;

	// Load input and output; Both operations will be stored in place
	matrix32f_t input1[2], output1[2];
	input1[0].d = NULL; output1[0].d = NULL;
	input1[1].d = NULL; output1[2].d = NULL;

	// Create weight matrices
	matrix32f_t w_matrix[8];
	for(int i = 0; i < 8; i++) { w_matrix[i].d = NULL; }

	printf("Loading input vector (%s)...", input_path[0]);
	startClock();
	if(test = matrixFromCSV(input_path[0], matrix_dims[0], matrix_dims[1], &input1[0])) {
		printf("\nError (%d): failed to import %s!\n\n", test, input_path[0]);
		ret = 3; goto exit;
	}
	printf("OK! (%.2f ms)\n", clockToMS(readClock()));

	printf("Loading output vector (%s)...", input_path[1]);
	startClock();
	if(test = matrixFromCSV(input_path[1], matrix_dims[2], matrix_dims[3], &output1[0])) {
		printf("\nError (%d): failed to import %s!\n\n", test, input_path[1]);
		ret = 3; goto exit;
	}
	printf("OK! (%.2f ms)\n", clockToMS(readClock()));

	// Create copies of input1 and output1
	newMatrix32f(matrix_dims[0], matrix_dims[1], &input1[1]);
	for(size_t i = 0; i < input1[0].h*input1[0].w; i++) { input1[1].d[i] = input1[0].d[i]; }

	newMatrix32f(matrix_dims[2], matrix_dims[3], &output1[1]);
	for(size_t i = 0; i < output1[0].h*output1[0].w; i++) { output1[1].d[i] = output1[0].d[i]; }

	// Load all input weights
	for(int channel = 0; channel < 4; channel++) {
		// Load input scale
		printf("Loading input scale parameters (%s)...", matrix_path[channel]);
		startClock();
		if(test = matrixFromCSV(matrix_path[channel], matrix_dims[0], matrix_dims[1], &w_matrix[channel])) {
			printf("\nError (%d): failed to import %s!\n\n", test, matrix_path[channel]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load input mean
		printf("Loading input mean parameters (%s)...", matrix_path[4+channel]);
		startClock();
		if(test = matrixFromCSV(matrix_path[4+channel], matrix_dims[0], matrix_dims[1], &w_matrix[4+channel])) {
			printf("\nError (%d): failed to import %s!\n\n", test, matrix_path[4+channel]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));
	}

	matrix32f_t *mean_w, *scale_w;

	// Perform tests and time them
	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	clock_t start_time = clock();
	for(size_t iter = 0; iter < iterations; iter++) {
		// Select channel's weight depending on which iteration we are currently in
		scale_w = &w_matrix[iter / (iterations/4)];
		mean_w  = &w_matrix[iter / (iterations/4) + 4];

		// Start
		startClock();
		hadamardProduct(&input1[0], scale_w, NULL);
		hadamardProduct(&input1[1], scale_w, NULL);
		matrixSum(&input1[0], mean_w, NULL);
		matrixSum(&input1[1], mean_w, NULL);

		// Check timer
		float last_time = readClock();
		best_time  = (last_time < best_time)  ? last_time : best_time;
		worst_time = (last_time > worst_time) ? last_time : worst_time;

		// Check values?
	}
	clock_t end_time = clock();
	float mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (float)iterations;

	printf("\n\tInput Shift-Scale Results\n");
	printf("\t=====================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);
	printf("\t Best Time:  %4.1f us (%+4.1f us)\n", clockToMS(best_time)*1000.0,  clockToMS(best_time)*1000.0-mean_iter_time_us);
	printf("\t Worst Time: %4.1f us (%+4.1f us)\n", clockToMS(worst_time)*1000.0, clockToMS(worst_time)*1000.0-mean_iter_time_us);
	printf("\t=====================================\n\n");

	// Get ready for output test
	for(uint8_t m = 0; m < 8; m++) { deleteMatrix(&w_matrix[m]); }

	// Output Test - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	// Load all output weights
	for(int channel = 0; channel < 4; channel++) {
		// Load output scale
		printf("Loading output scale parameters (%s)...", matrix_path[8+channel]);
		startClock();
		if(test = matrixFromCSV(matrix_path[8+channel], matrix_dims[2], matrix_dims[3], &w_matrix[channel])) {
			printf("\nError (%d): failed to import %s!\n\n", test, matrix_path[8+channel]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load output mean
		printf("Loading output mean parameters (%s)...", matrix_path[12+channel]);
		startClock();
		if(test = matrixFromCSV(matrix_path[12+channel], matrix_dims[2], matrix_dims[3], &w_matrix[4+channel])) {
			printf("\nError (%d): failed to import %s!\n\n", test, matrix_path[12+channel]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));
	}

	// Perform tests and time them
	best_time  = (clock_t)9e18;
	worst_time = 0;
	start_time = clock();
	for(size_t iter = 0; iter < iterations; iter++) {
		// Select channel's weight depending on which iteration we are currently in
		scale_w = &w_matrix[iter / (iterations/4)];
		mean_w  = &w_matrix[iter / (iterations/4) + 4];

		// Start
		startClock();
		hadamardProduct(&output1[0], scale_w, NULL);
		hadamardProduct(&output1[1], scale_w, NULL);
		matrixSum(&output1[0], mean_w, NULL);
		matrixSum(&output1[1], mean_w, NULL);

		// Check timer
		float last_time = readClock();
		best_time  = (last_time < best_time)  ? last_time : best_time;
		worst_time = (last_time > worst_time) ? last_time : worst_time;

		// Check values?
	}
	end_time = clock();
	mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (float)iterations;

	printf("\n\tOutput Shift-Scale Results\n");
	printf("\t=====================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);
	printf("\t Best Time:  %4.1f us (%+4.1f us)\n", clockToMS(best_time)*1000.0,  clockToMS(best_time)*1000.0-mean_iter_time_us);
	printf("\t Worst Time: %4.1f us (%+4.1f us)\n", clockToMS(worst_time)*1000.0, clockToMS(worst_time)*1000.0-mean_iter_time_us);
	printf("\t=====================================\n\n");


exit:
	for(uint8_t m = 0; m < 8; m++) { deleteMatrix(&w_matrix[m]); }
	for(uint8_t m = 0; m < 2; m++) {
		deleteMatrix(&input1[m]);
		deleteMatrix(&output1[m]);
	}
	return ret;
}

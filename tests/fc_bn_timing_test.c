#include <stdio.h>
#include <string.h>

#include "csv.h"
#include "clock.h"
#include "matrix_math.h"
#include "lut.h"

static const char* const 	matrix_name[] = {"Fully Connected Layer Weights", "Batch Norm. Mean values", "Batch Norm. gamma/Var values", "Batch Norm. Beta values"};
static const char* const	matrix_path[] = {
	"parameters/csv/fc_bn_1/fc1_w_drums.csv", "parameters/csv/fc_bn_1/bn1_drums_mean.csv", "parameters/csv/fc_bn_1/bn1_drums_gv.csv", "parameters/csv/fc_bn_1/bn1_drums_beta.csv",
	"parameters/csv/fc_bn_2/fc2_w_drums.csv", "parameters/csv/fc_bn_2/bn2_drums_mean.csv", "parameters/csv/fc_bn_2/bn2_drums_gv.csv", "parameters/csv/fc_bn_2/bn2_drums_beta.csv",
	"parameters/csv/fc_bn_3/fc3_w_drums.csv", "parameters/csv/fc_bn_3/bn3_drums_mean.csv", "parameters/csv/fc_bn_3/bn3_drums_gv.csv", "parameters/csv/fc_bn_3/bn3_drums_beta.csv"
};
static const size_t			input_dim[]   = { 2974, 1024, 512 };
static const size_t 		matrix_dims[] = {
	2974, 512,		1, 512,		1, 512, 	1, 512,
	1024, 512,		1, 512,		1, 512, 	1, 512,
	512, 4098,		1, 4098,	1, 4098, 	1, 4098
};
static const char* const tanhlut_path = "lut/tanh257.lut";

static const char* const input_path[] = { "csv/test1x2974.csv", "csv/test1x1024.csv", "csv/test1x512-3.csv" } /* { "csv2/test1x512.csv", "csv2/test1x512-2.csv", "csv2/test1x4098.csv" }*/;
static const size_t path_idx[] = {0, 28, 30, 28};

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Fully Connected Layer - Batch Normalization Timing Test (all layers)");
#ifndef SERIAL
	printf(" (NEON)");
#endif
#ifdef DEBUG
	printf(" [Debug Build]");
#endif
	printf("\n");

	if(argc > 2) {
		printf("Usage: %s [iterations]\n\n", argv[0]);
		return 1;
	}

	// For loading messages
	setvbuf (stdout, NULL, _IONBF, BUFSIZ);

	// Get number of iterations or default to 16
	uint32_t iterations = (argc==2) ? atoi(argv[1]) : 16;

	// Load tanh LUT
	lut32f_t tanhlut;
	tanhlut.data = NULL;
	if(load32fLUT(&tanhlut, tanhlut_path)) {
		printf("Error: could not load %s.\n\n", tanhlut_path);
		return -1;
	}

	// Load input and make output
	matrix32f_t input1, output1;
	input1.d = NULL; output1.d = NULL;

	// Load weight matrices
	matrix32f_t fc_w_mat;
	matrix32f_t bn_mean_mat, bn_gammavar_mat, bn_beta_mat;

	fc_w_mat.d = NULL;
	bn_mean_mat.d = NULL; bn_gammavar_mat.d = NULL; bn_beta_mat.d = NULL;

	// Use a pointer array for quickly determining where to load what
	matrix32f_t *matrix_ptr[] = { &fc_w_mat, &bn_mean_mat, &bn_gammavar_mat, &bn_beta_mat };
	int8_t test;

	// We'll try all three layers
	for(uint8_t layer = 0; layer < 3; layer++) {
		printf("\n");

		for(uint8_t m = 0; m < 4; m++) {
			printf("Loading %s (%s)...", matrix_name[m], matrix_path[layer*4+m]);

			startClock();
			if(test = matrixFromCSV(matrix_path[layer*4+m], matrix_dims[layer*8 + 2*m], matrix_dims[layer*8 + 2*m+1], matrix_ptr[m])) {
				printf("\nError (%d): failed to import %s!\n\n", test, matrix_path[m]);
				ret = 3; goto exit;
			}
			printf("OK! (%.2f ms)\n", clockToMS(readClock()));
		}

		// Load vector to use as input
		startClock();
		printf("Loading input vector (%s)...", input_path[layer]);
		if(test = matrixFromCSV(input_path[layer], 1, input_dim[layer], &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, input_path[layer]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for the final output
		if(newMatrix32f(1, matrix_dims[layer*8+1], &output1)) {
			printf("Error: failed to create the final output matrix.\n\n");
			ret = -3; goto exit;
		}

		// Perform tests and time them
		clock_t fc_time = 0;
		clock_t best_time  = (clock_t)9e18;
		clock_t worst_time = 0;
		clock_t start_time = clock();
		for(size_t iter = 0; iter < iterations; iter++) {
			startClock();

			// Fully Connected Layer; after this operation all operations create 1x512 matrices
			multVecByMat(&input1, &fc_w_mat, &output1);
			fc_time += readClock();

			// Batch Normalization is just a series of elementwise, linear operations
			matrixDiff(&output1, &bn_mean_mat, &output1);
			hadamardProduct(&output1, &bn_gammavar_mat, NULL);
			matrixSum(&output1, &bn_beta_mat, NULL);

			// Activation function (tanh on l1, relu on l2 and none on l3)
			switch(layer) {
				case 0: // layer 1 => tanh
					clampingLUT(&output1, &tanhlut, NULL); break;
				case 1: // layer 2 => relu
					relu(&output1, NULL); break;
				default: // layer 3 => none
					break;
			}
			// Check timer
			float last_time = readClock();
			best_time  = (last_time < best_time)  ? last_time : best_time;
			worst_time = (last_time > worst_time) ? last_time : worst_time;

			// Check values?
		}
		clock_t end_time = clock();
		float mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (float)iterations;
		float mean_fc_time_ms = clockToMS(fc_time) / (float)iterations;

		printf("\n\tLayer %d Results\n", layer+1);
		printf("\t=====================================\n");
		printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
		printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);
		printf("\t Best Time:  %4.1f us (%+4.1f us)\n", clockToMS(best_time)*1000.0,  clockToMS(best_time)*1000.0-mean_iter_time_us);
		printf("\t Worst Time: %4.1f us (%+4.1f us)\n", clockToMS(worst_time)*1000.0, clockToMS(worst_time)*1000.0-mean_iter_time_us);
		printf("\t Mean FC Time/iter.:   %2.3f ms\n", mean_fc_time_ms);
		printf("\t=====================================\n\n");

		for(uint8_t m = 0; m < 4; m++) { deleteMatrix(matrix_ptr[m]); }
		deleteMatrix(&input1);
		deleteMatrix(&output1);
	}

exit:
	for(uint8_t m = 0; m < 4; m++) { deleteMatrix(matrix_ptr[m]); }
	deleteMatrix(&input1);
	deleteMatrix(&output1);
	return ret;
}

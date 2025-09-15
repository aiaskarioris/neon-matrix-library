#include <stdio.h>
#include <string.h>

#include "clock.h"
#include "csv.h"
#include "matrix_math.h"


int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Matrix Conversion to 8-bit Test");
#ifndef SERIAL
	printf(" (NEON)");
#endif
#ifdef DEBUG
	printf(" [Debug Build]");
#endif
	printf("\n\n");

	if(argc != 6 && argc != 5) {
		printf("Usage: %s [input width] [input height] [filename] [int bits] [iterations]\n\n", argv[0]);
		return 1;
	}

	// Get number of iterations or default to 16
	size_t inw = atoi(argv[1]);
	size_t inh = atoi(argv[2]);
	uint32_t iterations = atoi(argv[argc-1]);
	uint8_t int_bits = (argc == 6) ? atoi(argv[4]) : 4;
	printf("Integer bits: %d\n", int_bits);

	// Load input and make output
	matrix32f_t input0;
	matrix32f_t output0;
	input0.d = NULL;
	output0.d = NULL;

	// Our cool buffer
	int8_t *buffer8bit = NULL;

	int8_t test;
	printf("Loading %s...", argv[3]);
	startClock();
	if(test = matrixFromCSV(argv[3], inw, inh, &input0)) {
		printf("\nError (%d): failed to import %s!\n\n", test, argv[3]);
		ret = 3; goto exit;
	}
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));

	if(newMatrix32f(inw, inh, &output0)) {
		printf("Error: failed to create output matrix (%dx%d)!\n\n", inw,inh);
		ret = 40; goto exit;
	}

	buffer8bit = (int8_t*)malloc(inw*inh);
	if(!buffer8bit){
		printf("Error: failed to allocate memory for 8-bit buffer.\n\n");
		ret = 50; goto exit;
	}

	/*
	// Show output
	printf("Input:");
	for(size_t i = 0; i < input0.w*input0.h; i++) {
		printf("%c%+01.6f", (i%16 == 0) ? '\n' : ' ', input0.d[i]);
	} printf("\n\n");
	*/
	// Perform tests and time them
	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	uint32_t best_time_idx = -1, worst_time_idx = -1;
	clock_t middle_time = 0;


	clock_t start_time = clock();
	for(size_t iter = 0; iter < iterations; iter++) {
		startClock();

		// Convert matrix to quantized 8-bit values
		dumpFloat32to8bit(&input0, int_bits, buffer8bit);

		middle_time += readClock();

		// Convert back
		matrixFrom8bit(buffer8bit, int_bits, &input0);

	}
	clock_t end_time = clock();
	float mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (float)iterations;
	float mean_middle_time  = clockToMS(middle_time) * 1000.0 / (float)iterations;

	/*
	printf("8-bit:");
	for(size_t i = 0; i < input0.w*input0.h; i++) {
		printf("%c%3d", (i%16 == 0) ? '\n' : ' ', buffer8bit[i]);
	} printf("\n\n");

	// Show output
	printf("Output:");
	for(size_t i = 0; i < input0.w*input0.h; i++) {
		printf("%c%+01.6f", (i%16 == 0) ? '\n' : ' ', input0.d[i]);
	} printf("\n\n");
	*/

	printf("\n\tResults\n");
	printf("\t=====================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);
	// printf("\t Best Time:  %4.1f us (%+4.1f us, iter. #%d)\n", clockToMS(best_time)*1000.0,  clockToMS(best_time)*1000.0-mean_iter_time_us, best_time_idx);
	// printf("\t Worst Time: %4.1f us (%+4.1f us, iter. #%d)\n", clockToMS(worst_time)*1000.0, clockToMS(worst_time)*1000.0-mean_iter_time_us, worst_time_idx);
	printf("\n\t 32-bit -> 8-bit Mean Time: %4.1f us\n", mean_middle_time);
	printf("\t 32-bit <- 8-bit Mean Time: %4.1f us\n", mean_iter_time_us-mean_middle_time);
	printf("\t=====================================\n\n");

exit:
	deleteMatrix(&input0);
	deleteMatrix(&output0);
	if(buffer8bit) { free(buffer8bit); }
	return ret;
}

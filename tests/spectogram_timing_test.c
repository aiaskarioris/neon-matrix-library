#include <stdio.h>
#include <string.h>


#include "clock.h"
#include "csv.h"
#include "lut.h"
#include "matrix_math.h"


static const char* const input_path[]  = { "csv/test1x4098-re.csv", "csv/test1x4098-im.csv" };
static const char* const sqrt_lut_path = "lut/sqrt65536.lut";

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Spectogram Calculation Timing Test (NEON)");
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

	// LUT
	lut32f_t sqrt_lut;
	sqrt_lut.data = NULL;

	// Load input and make output
	matrix32f_t input_real, input_imag;
	matrix32f_t output_real, output_imag, output0;
	input_real.d = NULL; input_imag.d = NULL;
	output_real.d = NULL; output_imag.d = NULL;
	output0.d = NULL;

	int8_t test;
	printf("Loading %s (%s)...", "real input", input_path[0]);
	startClock();
	if(test = matrixFromCSV(input_path[0], 1, 4098, &input_real)) {
		printf("\nError (%d): failed to import %s!\n\n", test, input_path[0]);
		ret = 3; goto exit;
	}
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));

	printf("Loading %s (%s)...", "imaginery input", input_path[1]);
	startClock();
	if(test = matrixFromCSV(input_path[1], 1, 4098, &input_imag)) {
		printf("\nError (%d): failed to import %s!\n\n", test, input_path[1]);
		ret = 3; goto exit;
	}
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));

	if(newMatrix32f(1, 4098, &output0)) {
		printf("Error: failed to create output matrix (%dx%d)!\n\n", 1, 4098);
		ret = 40; goto exit;
	}

	if(newMatrix32f(1, input_real.w, &output_real) || newMatrix32f(1, input_imag.w, &output_imag)) {
		printf("Error: failed to create output matrices for inputs!\n\n");
		ret = 40; goto exit;
	}

	// Load square root LUT
	if(load32fLUT(&sqrt_lut, sqrt_lut_path)) {
		printf("Error: failed to load square root LUT.\n\n");
		ret = 5; goto exit;
	}

	// Perform tests and time them
	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	uint32_t best_time_idx = -1, worst_time_idx = -1;

	clock_t start_time = clock();
	for(size_t iter = 0; iter < iterations; iter++) {
		startClock();

		// Raise inputs to the power of 2
		// NOTE: In reality, we would do this with 2 threads (?)
		elementwisePow2(&input_real, &output_real);
		elementwisePow2(&input_imag, &output_imag);
		matrixSum(&output_real, &output_imag, &output0);
		sqrtLUT(&output0, &sqrt_lut, &output0);

		// Check timer
		float last_time = readClock();
		best_time_idx 	= (last_time < best_time)  ? iter : best_time_idx;
		worst_time_idx 	= (last_time > worst_time) ? iter : worst_time_idx;
		best_time  		= (last_time < best_time)  ? last_time : best_time;
		worst_time 		= (last_time > worst_time) ? last_time : worst_time;

		// Check values?
	}
	clock_t end_time = clock();
	float mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (float)iterations;

	printf("\n\tResults\n");
	printf("\t=====================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);
	printf("\t Best Time:  %4.1f us (%+4.1f us, iter. #%d)\n", clockToMS(best_time)*1000.0,  clockToMS(best_time)*1000.0-mean_iter_time_us, best_time_idx);
	printf("\t Worst Time: %4.1f us (%+4.1f us, iter. #%d)\n", clockToMS(worst_time)*1000.0, clockToMS(worst_time)*1000.0-mean_iter_time_us, worst_time_idx);
	printf("\t=====================================\n\n");

exit:
	deleteLUT32f(&sqrt_lut);
	deleteMatrix(&input_real);
	deleteMatrix(&input_imag);
	deleteMatrix(&output0);
	return ret;
}

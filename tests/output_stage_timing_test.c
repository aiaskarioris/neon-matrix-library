#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>

#include "clock.h"
#include "csv.h"
#include "stft.h"
#include "matrix_math.h"

static const char* const input_path = "csv/test1x4098.csv";

void printBlock(size_t len, float32_t *d);

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Wiener Filter and iFFT Test");
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

	// LUT
	lut32f_t atan_lut, sin_lut, cos_lut;
	atan_lut.data = NULL;
	sin_lut.data = NULL;
	cos_lut.data = NULL;

	// Create input/output matrices
	matrix32c_t fft_matrix; 	// STFT frame from original input
	matrix32f_t mask_estimate;	// Output of network for one channel
	matrix32f_t angles;			// Results of angle(fft_matrix)
	matrix32c_t fft_in;			// Modified STFT frame that will create audio_output
	matrix32f_t audio_output;	// Final audio output

	fft_matrix.d = NULL; mask_estimate.d = NULL; audio_output.d = NULL;
	angles.d = NULL; fft_in.d = NULL;



	// Load input
	size_t bin_count = 4096/2+1;
	int8_t test;
	printf("Loading DFT results from input (%s)...", input_path);
	startClock();
	if(test = matrixFromCSV(input_path, 1, bin_count*2, (matrix32f_t*)&fft_matrix)) {
		printf("\nError (%d): failed to import %s!\n\n", test, input_path);
		ret = 30; goto exit;
	}
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));

	// Create matrices
	if(newMatrix32f(1, bin_count, &mask_estimate)) {
		printf("Error: failed to create mask_estimate.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32c(1, bin_count, &fft_matrix)) {
		printf("Error: failed to create matrix for FFT output.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32f(1, 4096, &audio_output)) {
		printf("Error: failed to create audio output.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32f(1, bin_count, &angles)) {
		printf("Error: failed to create angles buffer.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32c(1, bin_count, &fft_in)) {
		printf("Error: failed to create buffer of final FFT.\n");
		ret = 40; goto exit;
	}

	// Load LUTs
	printf("Loading atan LUT...");
	if(test = load32fLUT(&atan_lut, "lut/atan65537.lut")) {
		printf("Error (%d\n)", test);
	}
	printf("OK\n");

	printf("Loading sine LUT...");
	if(test = load32fLUT(&sin_lut, "lut/sin256.lut")) {
		printf("Error (%d\n)", test);
	}
	printf("OK\n");
	printf("Loading cosine LUT...");
	if(test = load32fLUT(&cos_lut, "lut/cos256.lut")) {
		printf("Error (%d\n)", test);
	}
	printf("OK\n");


	// Initialize FFTW
	printf("Generating FFTW plan...");
	startClock();
#ifdef USE_THREADS
	int threads = 1;
	printf("(using %d threads) ", threads);
	fftwf_init_threads();
	fftwf_plan_with_nthreads(threads);
#endif
	fftwf_plan const plan = fftwf_plan_dft_c2r_1d(4096, fft_in.d, audio_output.d, FFTW_ESTIMATE);
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));

	// Perform tests and time them
	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	uint32_t best_time_idx = -1, worst_time_idx = -1;

	clock_t start_time = clock();
	clock_t fft_time = 0;
	for(size_t iter = 0; iter < iterations; iter++) {
		startClock();
		// Apply filter
		angleLUT_c(&fft_matrix, &atan_lut, &angles);
		expiLUT(&angles, &sin_lut, &cos_lut, &fft_in);
		hadamardProduct_cbr(&fft_in, &mask_estimate, NULL);

		// iFFT
		clock_t fftstart = clock();
		fftwf_execute(plan);
		fft_time += clock() - fftstart;

		// Check timer
		float last_time = readClock();
		best_time_idx 	= (last_time < best_time)  ? iter : best_time_idx;
		worst_time_idx 	= (last_time > worst_time) ? iter : worst_time_idx;
		best_time  		= (last_time < best_time)  ? last_time : best_time;
		worst_time 		= (last_time > worst_time) ? last_time : worst_time;
	}
	clock_t end_time = clock();
	float mean_iter_time_us = clockToMS(end_time - start_time) * 1000.0 / (float)iterations;

	printf("\n\tResults\n");
	printf("\t=====================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f us\n", mean_iter_time_us);
	printf("\t Best Time:  %4.1f us (%+4.1f us, iter. #%d)\n", clockToMS(best_time)*1000.0,  clockToMS(best_time)*1000.0-mean_iter_time_us, best_time_idx);
	printf("\t Worst Time: %4.1f us (%+4.1f us, iter. #%d)\n", clockToMS(worst_time)*1000.0, clockToMS(worst_time)*1000.0-mean_iter_time_us, worst_time_idx);
	printf("\t iFFT Mean Time: %4.1f us\n", clockToMS(fft_time/(float)iterations)*1000.0);
	printf("\t=====================================\n\n");

exit:
	if(ret < 10)
		fftwf_destroy_plan(plan);

	deleteLUT32f(&atan_lut);
	deleteLUT32f(&sin_lut);
	deleteLUT32f(&cos_lut);

	deleteMatrix((matrix32f_t*)&fft_matrix);
	deleteMatrix(&mask_estimate);
	deleteMatrix(&angles);
	deleteMatrix((matrix32f_t*)&fft_in);
	deleteMatrix(&audio_output);

	return ret;
}

void printBlock(size_t len, float32_t *d) {
	printf("%08x", d);
	for(size_t i = 0; i < len; i++) {
		if(i%32==0) { printf("\n%04d: ", i); }
		printf("%+05.1f ", d[i]);
	}
	printf("\n");
}

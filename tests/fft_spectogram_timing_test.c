#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>

#include "clock.h"
#include "csv.h"
#include "stft.h"
#include "matrix_math.h"

static const char* const input_path = "csv/audio1x2048.csv";
static const char* const sqrt_lut_path = "lut/sqrt24bits.lut";

void printBlock(size_t len, float32_t *d);

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("FFT and Spectogram Calculation Test");
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
	lut32f_t sqrt_lut;
	sqrt_lut.data = NULL;

	// Create matrices for calculations
	matrix32f_t audio_input;
	matrix32f_t audio_input_extended;
	matrix32c_t fft_matrix;
	matrix32f_t spec_output;
	matrix32f_t hann_window;

	audio_input.d = NULL;
	audio_input_extended.d = NULL;
	fft_matrix.d  = NULL;
	spec_output.d = NULL;
	hann_window.d = NULL;

	// Load input
	int8_t test;
	printf("Loading audio input (%s)...", input_path);
	startClock();
	if(test = matrixFromCSV(input_path, 1, 2048, &audio_input)) {
		printf("\nError (%d): failed to import %s!\n\n", test, input_path);
		ret = 30; goto exit;
	}
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));

	// Create matrices
	if(newMatrix32f(1, 4096, &audio_input_extended)) {
		printf("Error: failed to create Hann Window.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32c(1, 4096/2+1, &fft_matrix)) {
		printf("Error: failed to create matrix for FFT output.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32f(1, 4096, &hann_window)) {
		printf("Error: failed to create Hann Window.\n");
		ret = 40; goto exit;
	}

	if(newMatrix32f(1, 4096/2+1, &spec_output)) {
		printf("Error: failed to create matrix for output spectogram.\n");
		ret = 40; goto exit;
	}

	// Initialize FFTW
	printf("Generating FFTW plan...");
	startClock();
#ifdef USE_THREADS
	int threads = 1;
	printf("(using %d threads) ", threads);
	fftwf_init_threads();
	fftwf_plan_with_nthreads(threads);
#endif
	fftwf_plan const plan = fftwf_plan_dft_r2c_1d(4096, audio_input_extended.d, fft_matrix.d, FFTW_ESTIMATE);
	printf("OK!\t(%.2f ms)\n", clockToMS(readClock()));
	//fftwf_print_plan(plan);


	// Load square root LUT
	if(load32fLUT(&sqrt_lut, sqrt_lut_path)) {
		printf("Error: failed to load square root LUT.\n\n");
		ret = 4; goto exit;
	}

	// Generate Hann Window
	hannWindow(4096, &hann_window);


	// Perform tests and time them
	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	uint32_t best_time_idx = -1, worst_time_idx = -1;

	clock_t start_time = clock();
	clock_t temp_time;
	clock_t extension_time = 0, hann_time = 0, fft_time = 0, spectogram_time = 0;
	for(size_t iter = 0; iter < iterations; iter++) {
		startClock();
		// Prepate audio buffer for FFT
		temp_time = clock();
		extendInput(&audio_input, &audio_input_extended, 2);
		extension_time += clock() - temp_time;

		temp_time = clock();
		hadamardProduct(&audio_input_extended, &hann_window, NULL);
		hann_time += clock() - temp_time;

		// FFT
		temp_time = clock();
		fftwf_execute(plan);
		fft_time += clock() - temp_time;

		// FFT to spectogram
		temp_time = clock();
		fftToSpectogram(&fft_matrix, &spec_output, &sqrt_lut);
		spectogram_time += clock() - temp_time;

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
	printf("\t Extension Mean Time: %4.1f us\n", clockToMS(extension_time/(float)iterations)*1000.0);
	printf("\t Hann Window Mean Time: %4.1f us\n", clockToMS(hann_time/(float)iterations)*1000.0);
	printf("\t FFTW Mean Time: %4.1f us\n", clockToMS(fft_time/(float)iterations)*1000.0);
	printf("\t Spec. Mean Time: %4.1f us\n", clockToMS(spectogram_time/(float)iterations)*1000.0);
	printf("\t=====================================\n\n");

exit:
	if(ret < 10)
		fftwf_destroy_plan(plan);

	deleteLUT32f(&sqrt_lut);
	deleteMatrix(&audio_input);
	deleteMatrix(&spec_output);
	deleteMatrix(&hann_window);
	deleteMatrix((matrix32f_t*)&fft_matrix);
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

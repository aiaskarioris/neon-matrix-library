#include <stdio.h>
#include <string.h>

#include "lstm.h"
#include "csv.h"
#include "clock.h"

const char *frame_in_path[] = { "csv/frame1.csv", "csv/frame2.csv", "csv/frame3.csv" };
const char *param_path[] = {
	"parameters/csv/lstm_drums_wl0/lstm_drums_wf.csv", "parameters/csv/lstm_drums_wl0/lstm_drums_wc.csv",
	"parameters/csv/lstm_drums_wl0/lstm_drums_wi.csv", "parameters/csv/lstm_drums_wl0/lstm_drums_wo.csv",

	"parameters/csv/lstm_drums_wl0/lstm_drums_uf.csv", "parameters/csv/lstm_drums_wl0/lstm_drums_uc.csv",
	"parameters/csv/lstm_drums_wl0/lstm_drums_ui.csv", "parameters/csv/lstm_drums_wl0/lstm_drums_uo.csv",

	"parameters/csv/lstm_drums_wl0/lstm_drums_fbias.csv", "parameters/csv/lstm_drums_wl0/lstm_drums_cbias.csv",
	"parameters/csv/lstm_drums_wl0/lstm_drums_ibias.csv", "parameters/csv/lstm_drums_wl0/lstm_drums_obias.csv",
};

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("LSTM Timing Test");
#ifndef SERIAL
	printf(" (NEON)");
#endif
#ifdef DEBUG
	printf(" [Debug Build]");
#endif
	printf("\n\n");

	if(argc == 1 || argc > 3) {
		printf("Usage: %s [contex-size] [iterations]\n\n", argv[0]);
		return 1;
	}

	// For loading messages
	setvbuf (stdout, NULL, _IONBF, BUFSIZ);

	// If no argument is passed, both contex-size and iterations are assumed
	// If one argument is passed, it is interpreted as the context-size and iterations are assumed
	uint32_t ctx_size   = atoi(argv[1]);
	uint32_t iterations = (argc == 3) ? atoi(argv[2]) : 1024;

	// Load input and make output
	matrix32f_t *finput;
	matrix32f_t *foutput;

	finput  = (matrix32f_t*)malloc(ctx_size * sizeof(matrix32f_t));
	foutput = (matrix32f_t*)malloc(ctx_size * sizeof(matrix32f_t));
	for(uint32_t i = 0; i < ctx_size; i++) {
		// We'll fill input buffers later
		finput[i].d = NULL;

		// Allocate output buffers
		newMatrix32f(1, 512, &foutput[i]);
	}

	// Create these guys
	lstm_t lstm_f[3];
	lstm_t lstm_b[3];

	lut32f_t sigmoid_lut, tanh_lut;
	sigmoid_lut.data = NULL; tanh_lut.data = NULL;

	// Create lstms
	for(int i = 0; i < 3; i++){
		printf("\r[%d/6] Created fLSTM Cell %d", i*2, i);
		lstmCreate(512, 256, 0, &lstm_f[i]);
		printf("\r[%d/6] Created bLSTM Cell %d", i*2+1, i);
		lstmCreate(512, 256, 1, &lstm_b[i]);
	}
	printf("\r[6/6] Created all LSTM Cells.\n");

	// Load LUTs
	printf("Loading sigmoid LUT...");
	if(load32fLUT(&sigmoid_lut, "lut/sigmoid.lut")){
		printf("Error: Could not load LUT for sigmoid function.\n");
		ret = 3; goto exit;
	}
	printf("OK\n");

	printf("Loading tanh LUT...");
	if(load32fLUT(&tanh_lut, "lut/tanh.lut")){
		printf("Error: Could not load LUT for tanh function.\n");
		ret = 3; goto exit;
	}
	printf("OK\n");

	// LUTs are ready; configure lstms
	for(int i = 0; i < 3; i++) {
		lstmSetLUTs(&sigmoid_lut, &tanh_lut, &lstm_f[i]);
		lstmSetLUTs(&sigmoid_lut, &tanh_lut, &lstm_b[i]);
	}

	// Set up parameters; We'll use the same numbers for all cells
	for(int i = 0; i < 3; i++) {
		lstmLoadParameters(param_path, &lstm_f[i]);
		printf("\r[%d/6] Loading parameters...", i*2);
		lstmLoadParameters(param_path, &lstm_b[i]);
		printf("\r[%d/6] Loading parameters...", i*2+1);
	}
	printf("\r[6/6] All LSTM parameters have been loaded.\n");

	// Fill input buffers
	int i;
	for(i = 0; i < 3; i++) {
		printf("\r[%d/%d] Populating input buffers (importing)...", i, ctx_size);
		if(matrixFromCSV(frame_in_path[i], 1, 512, &finput[i])) {
			printf("Error: Could not load input frame (i: %d, path: %s).\n", i, frame_in_path[i]);
			ret = 5; goto exit;
		}
	}
	for(i; i < ctx_size; i++) {
		printf("\r[%d/%d] Populating input buffers (copying)...  ", i, ctx_size);
		if(newMatrix32f(1, 512, &finput[i])) {
			printf("Error: Could not allocate memory for input buffer.\n");
			ret = 6; goto exit;
		}
		// Copy data from one of the first 3 input buffers
		memcpy(finput[i].d, finput[i%3].d, sizeof(float32_t) * finput[0].w*finput[0].h);
	}
	printf("\r[%d/%d] Input buffers ready: 3 imported, %d copied.\n", ctx_size, ctx_size, ctx_size-3);

	// Connect LSTM cells
	lstmConnect(&lstm_f[1], &lstm_f[0], &lstm_b[0]);
	lstmConnect(&lstm_f[2], &lstm_f[1], &lstm_b[1]);
	lstmConnect(&lstm_b[1], &lstm_b[0], &lstm_f[0]);
	lstmConnect(&lstm_b[2], &lstm_b[1], &lstm_f[1]);

	// Perform tests and time them
	clock_t best_time  = (clock_t)9e18;
	clock_t worst_time = 0;
	clock_t start_time = clock();
	clock_t lstm_mean_time = 0;

	// This is a simple test; All output Hs are copied
	// NOTE: This is NOT how the final design will work
	for(size_t iter = 0; iter < iterations; iter++) {
		startClock();

		clock_t lstm_time = clock();
		for(size_t c = 0; c < ctx_size; c++) {
			lstm_in(&finput[c], &lstm_f[0]);
			lstm_in(&finput[ctx_size - c - 1], &lstm_b[0]);

			lstm_mid(&lstm_f[1]);
			lstm_mid(&lstm_b[1]);

			lstm_out(&lstm_f[2], &foutput[c]);
			lstm_out(&lstm_b[2], &foutput[ctx_size - c - 1]);
		}
		lstm_mean_time += clock() - lstm_time;

/*
		// Copy outputs to output buffer
		clock_t copy_time = clock();
		for(size_t c = 0; c < ctx_size; c++) {
			matrixConcat(&buffer[c], &buffer[ctx_size - c - 1], &foutput[c]);
		}
		copy_time = clock() - copy_time;
*/
		// Check timer
		// float last_time = readClock();
		// best_time  = (last_time < best_time)  ? last_time : best_time;
		// worst_time = (last_time > worst_time) ? last_time : worst_time;
	}

	clock_t end_time = clock();
	float mean_iter_time_ms = clockToMS(end_time - start_time) / (float)iterations;

	printf("\nLSTM Results\n");
	printf("\t=====================================\n");
	printf("\t Time for %4d iterations: %4.3f ms\n", iterations, clockToMS(end_time - start_time));
	printf("\t Mean Time/iter.:   %2.2f ms\n", mean_iter_time_ms);
	// printf("\t Best Time:  %4.1f ms (%+4.1f ms)\n", clockToMS(best_time),  clockToMS(best_time)-mean_iter_time_ms);
	// printf("\t Worst Time: %4.1f ms (%+4.1f ms)\n", clockToMS(worst_time), clockToMS(worst_time)-mean_iter_time_ms);
	printf("\t LSTM Time: %4.1f ms\n", clockToMS((float)lstm_mean_time / (float)iterations));
	//printf("\t Copy Time: %4.1f ms\n", clockToMS(copy_time));
	printf("\t=====================================\n\n");


exit:
	for(uint8_t m = 0; m < 3; m++) {
		lstmDelete(&lstm_f[m]);
		lstmDelete(&lstm_b[m]);
	}
	deleteLUT32f(&sigmoid_lut);
	deleteLUT32f(&tanh_lut);

	for(uint32_t i = 0; i < ctx_size; i++) {
		deleteMatrix(&finput[i]);
		deleteMatrix(&foutput[i]);
	}
	free(finput);
	free(foutput);

	return ret;
}

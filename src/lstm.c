#ifdef DEBUG
#include <stdio.h>
#endif

#include <string.h> // memcpy

#include "lstm.h"
#include "csv.h"

// Calculates all gates and outputs of an LSTM cell
inline void lstm_process(matrix32f_t *input, lstm_t *lstm);
// This function is private; it is not available outside 'lstm.c'

// Initializes an LSTM cell, allocating the appropriate memory
// Depending on the layer of the LSTM, additional operations will be required before `lstm` will be used
int lstmCreate(size_t input_size, size_t hidden_size, uint8_t dir, lstm_t *lstm) {
	lstm->input_size  = input_size;
	lstm->hidden_size = hidden_size;
	lstm->direction = dir;

	// We'll create an array of matrix32f_t pointers to initialize; All matrices are
	// vectors of `input_size` length
	matrix32f_t* mat_to_init[] = {
		/* Internal */ 		&(lstm->c), &(lstm->h),
		/* Scratchpad */ 	&(lstm->f_scratchpad), &(lstm->c_scratchpad),
							&(lstm->i_scratchpad), &(lstm->o_scratchpad),
		/* General Purp. */ &(lstm->gp_scratchpad)
	};


	// Init matrices
	size_t mat_w;
	for(size_t i = 0; i < 7; i++) {
		// Middle and output cells concatenate their inputs into `gp_scratchpad` so it
		// should be equal to `input_size`. We will change gp_scratchpad->w to `hidden_size` when required
		// to "hide" extra memory from matrix_math functions when required.
		// NOTE: Since 2 out of 6 cells don't need a large gp_scratchpad, 512*2 floats or 2KiB are unused.
		mat_w = (i != 6) ? hidden_size : input_size;
		if(newMatrix32f(1, mat_w, mat_to_init[i])) {
#ifdef DEBUG
			printf("Error in create_lstm: Failed to allocate memory (matrix %d).\n", i);
#endif
			return 1;
		}

		// Hide extra memory from other functions
		lstm->gp_scratchpad.w = hidden_size;
		// note: `lstm_mid` or `lstm_out` will change this value for their runtime and revert it back
		// to `hidden_size` before returning. Since `free()` is used for de-allocation, the width value
		// can be safely changed to any arbitrary value.
	}

	// Clear both internal matrices
	clearMatrix(&(lstm->c));
	clearMatrix(&(lstm->h));

	// Set these pointers to something "safe"
	lstm->h_in0_ptr = NULL;
	lstm->h_in1_ptr = NULL;
	lstm->sigmoid_lut_ptr = NULL;
	lstm->tanh_lut_ptr = NULL;

	// Same for parameters/biases
	lstm->f_w.d = NULL; lstm->c_w.d = NULL;
	lstm->i_w.d = NULL; lstm->o_w.d = NULL;

	lstm->f_u.d = NULL; lstm->c_u.d = NULL;
	lstm->i_u.d = NULL; lstm->o_u.d = NULL;

	lstm->f_bias.d = NULL; lstm->c_bias.d = NULL;
	lstm->i_bias.d = NULL; lstm->o_bias.d = NULL;
	return 0;
}

void lstmSetLUTs(lut32f_t *sigmoid_lut, lut32f_t *tanh_lut, lstm_t *lstm) {
	// Store LUT Pointers
	lstm->sigmoid_lut_ptr 	= sigmoid_lut;
	lstm->tanh_lut_ptr 		= tanh_lut;
}

int lstmLoadParameters(const char **param_paths, lstm_t *lstm){
	// Create a pointer array
	matrix32f_t * const param_mat[] = {
		&lstm->f_w, &lstm->c_w, &lstm->i_w, &lstm->o_w,
		&lstm->f_u, &lstm->c_u, &lstm->i_u, &lstm->o_u,
		&lstm->f_bias, &lstm->c_bias, &lstm->i_bias, &lstm->o_bias
	};

	size_t h;
	int test;
	for(int i = 0; i < 12; i++) {
		if(i < 4)		{ h = lstm->input_size; }
		else if(i < 8) 	{ h = lstm->hidden_size;}
		else			{ h = 1; }

		if(test = matrixFromCSV(param_paths[i], h, lstm->hidden_size, param_mat[i])) {
#ifdef DEBUG
			printf("Error in lstmLoadParameters: Failed to load matrix #%d, function returned: %d.\n", i, test);
#endif
			return test;
		}
	}
}

// Frees memory of an LSTM Cell
void lstmDelete(lstm_t *lstm) {
	matrix32f_t* mat_to_del[] = {
		/* Internal */ 		&lstm->c, &lstm->h,
		/* Scratchpad */ 	&lstm->f_scratchpad, &lstm->c_scratchpad,
							&lstm->i_scratchpad, &lstm->o_scratchpad,
		/* General Purp. */ &lstm->gp_scratchpad
	};
	for(uint8_t i = 0; i < 7; i++) { deleteMatrix(mat_to_del[i]); }
}

// Configures `lstm0` to use `lstm_in0`'s and `lstm_in1`'s Hs as inputs
void lstmConnect(lstm_t *lstm0, lstm_t *lstm_in0, lstm_t *lstm_in1) {
#ifdef DEBUG
	if(lstm0 == NULL) { printf("Error in lstmConnect: Configuration target is uninitialized (lstm0 == NULL)\n"); return; }
	if((lstm_in0 == NULL) || (lstm_in1 == NULL)) { printf("Error in lstmConnect: Attempting to connect LSTM's whose Hs are NULL.\n"); return; }
#endif
	lstm0->h_in0_ptr = &lstm_in0->h;
	lstm0->h_in1_ptr = &lstm_in1->h;
}

// Executes code for LSTMs of input layer (layer 0)
void lstm_in(matrix32f_t *input, lstm_t *lstm) {
#ifdef DEBUG
	if(lstm == NULL) { printf("Error in lstm_in: lstm == NULL\n"); return; }
	if(lstm->h.d == NULL || lstm->c.d == NULL) { printf("Error in lstm_in: lstm->h->d == NULL || lstm->c->d == NULL\n"); return; }
	if((lstm->h_in0_ptr != NULL) || (lstm->h_in1_ptr != NULL)) { printf("Warning in lstm_in: h_in0/1 are not NULL.\n"); return; }
#endif

	// Calculate all gates
	lstm_process(input, lstm);
}

void lstm_mid(lstm_t *lstm) {
#ifdef DEBUG
	if(lstm == NULL) { printf("Error in lstm: lstm == NULL\n"); return; }
	if(lstm->h.d == NULL || lstm->c.d == NULL) { printf("Error in lstm: lstm->h->d == NULL || lstm->c->d == NULL\n"); return; }
	if((lstm->h_in0_ptr == NULL) || (lstm->h_in1_ptr == NULL)) { printf("Error in lstm: (lstm->h_in0 == NULL) || (lstm->h_in1 == NULL)\n"); return; }
#endif
	// Un-hide `gp_scratchpad` extra memory
	lstm->gp_scratchpad.w = lstm->input_size;

	// Create input matrix from `h_in0 and h_in1` `into gp_scratchpad`
	matrixConcat(lstm->h_in0_ptr, lstm->h_in1_ptr, &lstm->gp_scratchpad);

	// Do all calculations; Pass gp_scratchpad as the input
	lstm_process(&lstm->gp_scratchpad, lstm);

	// `gp_scratchpad` extra memory was hidden in lstm_process
}

void lstm_out(lstm_t *lstm, matrix32f_t *output){
#ifdef DEBUG
	if(lstm == NULL) { printf("Error in lstm_out: lstm == NULL\n"); return; }
	if(lstm->h.d == NULL || lstm->c.d == NULL) { printf("Error in lstm_out: lstm->h.d == NULL || lstm->c.d == NULL\n"); return; }
	if(output->d == NULL) { printf("Error in lstm_out: output->d == NULL\n"); return; }
	if((lstm->h_in0_ptr == NULL) || (lstm->h_in1_ptr == NULL)) { printf("Error in lstm_out: (lstm->h_in0 == NULL) || (lstm->h_in1 == NULL)\n"); return; }
#endif

	// Un-hide `gp_scratchpad` extra memory
	lstm->gp_scratchpad.w = lstm->input_size;

	// Create input matrix from `h_in0` and `h_in1` into `gp_scratchpad`
	matrixConcat(lstm->h_in0_ptr, lstm->h_in1_ptr, &lstm->gp_scratchpad);

	// Do all calculations; Pass gp_scratchpad as the input
	lstm_process(&lstm->gp_scratchpad, lstm);

	// H will be copied to the output
	size_t out_offset = (lstm->direction == 0) ? 0 : lstm->hidden_size;
	memcpy(output->d + out_offset, lstm->h.d, sizeof(float32_t) * lstm->hidden_size);
}

void lstm_process(matrix32f_t *input, lstm_t *lstm) {
	// Input * W is stored in `X_scratchpad`, depending on the gate.
	// Note that in some cases `gp_scratchpad` == `input` (arg); a
	// All Input multiplications should be completed before overwriting `gp_scratchpad`

	// Do input multiplications
	multVecByMat(input, &lstm->f_w, 	&lstm->f_scratchpad);
	multVecByMat(input, &lstm->c_w, 	&lstm->c_scratchpad);
	multVecByMat(input, &lstm->i_w, 	&lstm->i_scratchpad);
	multVecByMat(input, &lstm->o_w, 	&lstm->o_scratchpad);
	// (gp_scratchpad can be overwritten now)

	// Hide `gp_scratchpad` extra memory (see `lstmCreate`)
	// This line has no effect when executed from within `lstm_in`
	lstm->gp_scratchpad.w = lstm->hidden_size;

	matrix32f_t *gp_scratchpad = &lstm->gp_scratchpad;

	// Forget Gate
	multVecByMat(&lstm->h, 			&lstm->f_u, 	gp_scratchpad);
	matrixSum(&lstm->f_scratchpad,	gp_scratchpad, 	NULL); // (input * w) += (h * u)
	matrixSum(&lstm->f_scratchpad, 	&lstm->f_bias, 	NULL); // += bias
	clampingLUT(&lstm->f_scratchpad, lstm->sigmoid_lut_ptr, NULL);

	// Control Gate
	multVecByMat(&lstm->h, 			&lstm->c_u, 	gp_scratchpad);
	matrixSum(&lstm->c_scratchpad,	gp_scratchpad, 	NULL); // (input * w) += (h * u)
	matrixSum(&lstm->c_scratchpad, 	&lstm->c_bias, 	NULL); // += bias
	clampingLUT(&lstm->c_scratchpad, lstm->tanh_lut_ptr, NULL);

	// Input Gate
	multVecByMat(&lstm->h, 			&lstm->i_u, 	gp_scratchpad);
	matrixSum(&lstm->i_scratchpad,	gp_scratchpad, 	NULL); // (input * w) += (h * u)
	matrixSum(&lstm->i_scratchpad, 	&lstm->i_bias, 	NULL); // += bias
	clampingLUT(&lstm->i_scratchpad, lstm->tanh_lut_ptr, NULL);

	// Output Gate
	multVecByMat(&lstm->h, 			&lstm->o_u, 	gp_scratchpad);
	matrixSum(&lstm->o_scratchpad,	gp_scratchpad, 	NULL); // (input * w) += (h * u)
	matrixSum(&lstm->o_scratchpad, 	&lstm->o_bias, 	NULL); // += bias
	clampingLUT(&lstm->o_scratchpad, lstm->sigmoid_lut_ptr, NULL);

	// Update C and H
	// ct = ct-1 .* ft + it .* ct
	hadamardProduct(&lstm->i_scratchpad, &lstm->c_scratchpad, NULL); // it = it .* ct
	hadamardProduct(&lstm->c, &lstm->f_scratchpad, NULL); // ct = ft .* ct-1
	// here i_scratchpad and c are overwritten as to preserve memory
	matrixSum(&lstm->c, &lstm->i_scratchpad, NULL); // C is ready

	// ht = ot .* ct
	hadamardProduct(&lstm->o_scratchpad, &lstm->c, &lstm->h);
}


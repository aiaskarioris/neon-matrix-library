#pragma once

#include <arm_neon.h>

#include "matrix.h"
#include "matrix_math.h"
#include "lut.h"

typedef struct lstm_st {
	size_t 	input_size;
	size_t 	hidden_size;
	uint8_t direction;

	// Cell's Hold and Cell Matrices
	matrix32f_t h;
	matrix32f_t c;

	// Pointers' to other cells' H matrices; Used as input
	// Layer 0 LSTMs don't use these
	matrix32f_t *h_in0_ptr;
	matrix32f_t *h_in1_ptr;

	// Pointers to sigmoid and tanh LUTs (read-only)
	lut32f_t *sigmoid_lut_ptr;
	lut32f_t *tanh_lut_ptr;

	// Weights (for input)
	matrix32f_t f_w;
	matrix32f_t c_w;
	matrix32f_t i_w;
	matrix32f_t o_w;

	// Weights (for hold)
	matrix32f_t f_u;
	matrix32f_t c_u;
	matrix32f_t i_u;
	matrix32f_t o_u;

	// Biases
	matrix32f_t f_bias;
	matrix32f_t c_bias;
	matrix32f_t i_bias;
	matrix32f_t o_bias;

	// Scratchpad memory
	matrix32f_t f_scratchpad;
	matrix32f_t c_scratchpad;
	matrix32f_t i_scratchpad;
	matrix32f_t o_scratchpad;
	matrix32f_t gp_scratchpad; // general purpose

} lstm_t;

int  lstmCreate(size_t input_size, size_t hidden_size, uint8_t dir, lstm_t *lstm);
int  lstmLoadParameters(const char **param_paths, lstm_t *lstm);
void lstmSetLUTs(lut32f_t *sigmoid_lut, lut32f_t *tanh_lut, lstm_t *lstm);
void lstmDelete(lstm_t *lstm);
void lstmConnect(lstm_t *lstm0, lstm_t *lstm_in0, lstm_t *lstm_in1);

void lstm_in(matrix32f_t *input, lstm_t *lstm);
void lstm_mid(lstm_t *lstm);
void lstm_out(lstm_t *lstm, matrix32f_t *output);

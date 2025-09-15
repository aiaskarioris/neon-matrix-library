#pragma once
#include "matrix.h"

// This file contains functionality for Lookup tables as well as functions using Lookup-Tables

typedef struct lookuptable_f32_st {
	// Number of values within LUT
	uint32_t length;

	// Multiplication factor for getting input number into the address range
	float32_t mult_factor;
	// Number used to move the negative input space to a strictly-positive one.
	float32_t bias;

	float32_t *data;
} lut32f_t;

// Loads an lut32f_t object into `lut` from the file in `path`
uint8_t load32fLUT(lut32f_t *lut, const char *path);

void deleteLUT32f(lut32f_t *lut);


// Applies LUT to input with values exceeding LUT's borders clamped to the first/last LUT values.
// Used for tanh, sigmoid activation, etc
void clampingLUT(matrix32f_t *input0, lut32f_t *lut, matrix32f_t *output0);


// LUT function fitted to for square root.
// Expects non-negative input lesser than 200e3.
void sqrtLUT(matrix32f_t *input0, lut32f_t *lut, matrix32f_t *output0);

// Calculates the angle of a complex number using an atan lut
void angleLUT_c(matrix32c_t *input0, lut32f_t *lut, matrix32f_t *output0);

// For a real number `x`, calculates e^xi. (real input, imag. output)
// Utilizes a sine lut both for sine and cosine
void expiLUT(matrix32f_t *input0, lut32f_t *sinlut, lut32f_t *coslut, matrix32c_t *output0);

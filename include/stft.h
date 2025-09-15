#pragma once

#include "matrix.h" // NOTE: matrix.h includes <complex.h>, which should be defined before <fftw3.h> to enable the C99 complex type
#include <fftw3.h>

#include "lut.h"

typedef float complex complex_t;

// Struct holding settings and variables used for stft operations
typedef struct stft_st {
	// Input/Output pointers for STFT operation
	matrix32f_t *input_ptr;
	complex_t   *stft_out_ptr;

	// Settings for STFT
	uint32_t  fft_size;
	uint32_t  hop_size;

	// Weight buffer for input window
	matrix32f_t window_mat;

	// Struct for FFTW
	fftwf_plan plan;
} stft_t;

// Loads FFTW wisdom
//int initFFT(float32_t* inptr, complex_t* outptr, uint32_t hopsize, uint32_t fftsize, stft_t *settings);

// Loads a matrix with the Hann window used in STFT
void hannWindow(uint32_t fftsize, matrix32f_t *mat);

// Manipulates the input matrix so that it becomes longer
void extendInput(matrix32f_t *in0, matrix32f_t *out0, uint8_t rank);

// Converts FFTW Complex Output to matrix32f_t spectogram
void fftToSpectogram(matrix32c_t *fftin, matrix32f_t *out0, lut32f_t *sqrt_lut);

// Performs STFT and iSTFT operations using FFTW functions
void fft(stft_t *settings);
void ifft(stft_t *settings);



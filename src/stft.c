#include <math.h>
#include <string.h> // memcpy

#include "stft.h"
#include "matrix_math.h"

// Loads a matrix with the Hann window used in STFT
void hannWindow(uint32_t fftsize, matrix32f_t *mat) {
	const float pi = acosf(-1);
	for(size_t i = 0; i<fftsize; i++) { // win = sin(i*π/fftsize)^2
		mat->d[i] = sinf(i*pi/fftsize);
		mat->d[i] *= mat->d[i];
	}
}

#ifndef SERIAL
// NEON Code * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// Manipulates the input vector so that it becomes longer
void extendInput(matrix32f_t *in0, matrix32f_t *out0, uint8_t rank) {
	size_t in_len  = in0->w * in0->h;

#ifdef DEBUG
	if(rank != 2 && rank != 4 && rank != 8) { printf("Error in extendInput: Unsupported rank\n"); return ;}
	size_t out_len = out0->w * out0->h;
	if(in_len*rank != out_len) { printf("Error in extendInput: (in_len*rank != out_len)\n"); return; }
	if(in0->w != 1 && in0->h != 1) { printf("Error in extendInput: (in0->w != 1 && in0->h != 1)\n"); return; }
	if(out0->w != 1 && out0->h != 1) { printf("Error in extendInput: (out0->w != 1 && out0->h != 1)\n"); return; }
#endif
	// Copy input to output's first partition
	memcpy(out0->d, in0->d, in_len*sizeof(float32_t)); // memcpy(dest, src, num)

	// Copy input to the second partition of the output, flipped
	float32x4_t vreg;
	// `r` is used to index the buffer for writing; start from the last element of the 2nd quadr.
	size_t r = in_len + in_len - 4;
	for(size_t i = 0; i < in_len; i+=4) {
		vreg = vld1q_f32(&(in0->d[i]));
		// Flip vector's contents
		vreg = vrev64q_f32(vreg); // [a b c d] => [b a d c]; that's what vrev does
		vreg = vcombine_f32(vget_high_f32(vreg), vget_low_f32(vreg));
		vst1q_f32(&(out0->d[r]), vreg);
		r -= 4;
	}

	// Nothing more to do for doubling
	if(rank == 2) { return; }

	// Copy the first 2 quadrants to the 2 last
	memcpy(&out0->d[in_len*2], out0->d, in_len*2*sizeof(float32_t));
	if(rank == 4) { return; }

	// Quadruple 4 quadrants
	memcpy(&out0->d[in_len*4], out0->d, in_len*4*sizeof(float32_t));
}

// Converts FFTW Complex Output to matrix32f_t spectogram
void fftToSpectogram(matrix32c_t *fftin, matrix32f_t *out0, lut32f_t *sqrt_lut) {
	// Get squared magnitude
	squaredMagnitude(fftin, out0);

	// Get square root
	sqrtLUT(out0, sqrt_lut, NULL);
}

#else
// Serial Code (Non NEON) * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Manipulates the input matrix so that it becomes longer
void extendInput(matrix32f_t *in0, matrix32f_t *out0, uint8_t rank) {
	size_t in_len  = in0->w * in0->h;

#ifdef DEBUG
	if(rank != 2 && rank != 4 && rank != 8) { printf("Error in extendInput: Unsupported rank\n"); return ;}
	size_t out_len = out0->w * out0->h;
	if(in_len*rank != out_len) { printf("Error in extendInput: (in_len*rank != out_len)\n"); return; }
	if(in0->w != 1 && in0->h != 1) { printf("Error in extendInput: (in0->w != 1 && in0->h != 1)\n"); return; }
	if(out0->w != 1 && out0->h != 1) { printf("Error in extendInput: (out0->w != 1 && out0->h != 1)\n"); return; }
#endif
	// Copy input to output's first partition
	memcpy(out0->d, in0->d, in_len*sizeof(float32_t)); // memcpy(dest, src, num)

	// Copy input to the second partition of the output, flipped
	size_t r = in_len + in_len - 1;
	for(size_t i = 0; i < in_len; i+=1) {
		out0->d[r] = out0->d[i];
		r--;
	}

	// Nothing more to do for doubling
	if(rank == 2) { return; }

	// Copy the first 2 quadrants to the 2 last
	memcpy(&out0->d[in_len*2], out0->d, in_len*2*sizeof(float32_t));
	if(rank == 4) { return; }

	// Quadruple 4 quadrants
	memcpy(&out0->d[in_len*4], out0->d, in_len*4*sizeof(float32_t));
}

// Converts FFTW Complex Output to matrix32f_t spectogram
void fftToSpectogram(matrix32c_t *fftin, matrix32f_t *out0, lut32f_t *sqrt_lut) {
	// Power of 2
	elementwisePow2_complex(fftin);

	// Sum real part with imaginary
	size_t  len = fftin->w * fftin->h;
	size_t clen = len * 2;

	size_t i=0;

	for(size_t ci = 0; ci < len; ci+=2) {
		out0->d[i] = fftin->d[ci] + fftin->d[ci+1];
		i++;
	}

	// Get square root
	sqrtLUT(out0, sqrt_lut, NULL);
}
#endif



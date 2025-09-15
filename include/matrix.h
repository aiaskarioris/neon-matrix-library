#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <complex.h>
#include <arm_neon.h>

static const float fzero = 0.00;

// Quantization by multiplication is faster,
// more than 20% in certain cases.
#define QUANTIZE_BY_MULTIPLICATION
//#define QUANTIZE_BY_DIVISION

// Division factors used for reversion to float32_t.
// Essentially a LUT for 2^-(15-i)
// (7 and not 8 because one bit is used for the sign)
static const float quant_div_factor[] = {
    //Int. bits
    /*  0 */    3.051757812500000e-05,
    /*  1 */    6.103515625000000e-05,
    /*  2 */    1.220703125000000e-04,
    /*  3 */    2.441406250000000e-04,
    /*  4 */    4.882812500000000e-04,
    /*  5 */    9.765625000000000e-04,
    /*  6 */    1.953125000000000e-03,
    /*  7 */    3.906250000000000e-03,
    /*  8 */    7.812500000000000e-03,
    /*  9 */    1.562500000000000e-02,
    /* 10 */    3.125000000000000e-02,
    /* 11 */    6.250000000000000e-02,
    /* 12 */    1.250000000000000e-01,
    /* 13 */    2.500000000000000e-01,
    /* 14 */    5.000000000000000e-01,
    /* 15 */    1.000000000000000e+00
};

// Multiplication factors used for quantization
// quant_mul_factor = 1 / quant_div_factor, so that the
// factor must be multiplied with F32 inputs instead of dividing
static const float quant_mul_factor[] = {
    //Int. bits
    /*  0 */    3.276800000000000e+04,
    /*  1 */    1.638400000000000e+04,
    /*  2 */    8.192000000000000e+03,
    /*  3 */    4.096000000000000e+03,
    /*  4 */    2.048000000000000e+03,
    /*  5 */    1.024000000000000e+03,
    /*  6 */    5.120000000000000e+02,
    /*  7 */    2.560000000000000e+02,
    /*  8 */    1.280000000000000e+02,
    /*  9 */    6.400000000000000e+01,
    /* 10 */    3.200000000000000e+01,
    /* 11 */    1.600000000000000e+01,
    /* 12 */    8.000000000000000e+00,
    /* 13 */    4.000000000000000e+00,
    /* 14 */    2.000000000000000e+00,
    /* 15 */    1.000000000000000e+00
};

// Note: `quant_div_factor` can be multiplied with a quantized number to give the extended output
// and `quant_mul_factor` can be divide a quantized number to do the same.

typedef struct MATRIX32f_ST {
    size_t h; // width  (number of rows)
    size_t w; // height (number of colum)
    float32_t *d;
} matrix32f_t;

// Copy of matrix32f_t for complex numbers;
// This struct is essentially the same as a normal float matrix
// but with double the number of floats allocated for `d`
typedef struct MATRIX32C_ST {
    size_t h; // width  (number of rows)
    size_t w; // height (number of colum)
    float complex *d;
} matrix32c_t;

// Creates a new matrix object and allocates memory for it; 
// Returns non-zero on failure.
int newMatrix32f(size_t h, size_t w, matrix32f_t *mat);
int newMatrix32c(size_t h, size_t w, matrix32c_t *mat);

// De-Allocates memory for a matrix object
void deleteMatrix(matrix32f_t *mat);

// Sets contents of a matrix to zeros
void clearMatrix(matrix32f_t *mat);

// Flips the order of a matrix's contents; The input is always left intact
void flipVector(matrix32f_t *in0, matrix32f_t *out0);

// Concatenates in0 and in1 into out0; out0 is expected to have allocated memory
void matrixConcat(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0);

// Quantizes a 32-bit float matrix to 8-bit representation.
// `int_bits` specifies the number of bits used for the integer part. The result is always signed.
void dumpFloat32to8bit(matrix32f_t *mat, uint8_t int_bits, int8_t *dest);

// Converts 8-bit fixed point numbers to float32_t. Dimensions are read from `mat`,
// which should be already allocated.
void matrixFrom8bit(int8_t *src, uint32_t int_bits, matrix32f_t *mat);
void matrixFrom16bit(int16_t *src, uint32_t int_bits, matrix32f_t *mat);

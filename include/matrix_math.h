#pragma once
#include <math.h>
#include "matrix.h"

// This file contains declarations for linear algebra routines on matrices.
// For simplicity, a vector is also considered a matrix with one dimension set to 1.

// Matrix by Vector multiplication and vice-versa have reduced logic compared
// to a matrix multiplication.

// Vector by Matrix Multiplication; if `in0.h == 0` some loops can be skipped
void multVecByMat(matrix32f_t *vec0, matrix32f_t *mat1, matrix32f_t *out0);
// Vector by Matrix Multiplication; if `in1.h == 0` some loops can be skipped
void multMatByVec(matrix32f_t *mat0, matrix32f_t *vec1, matrix32f_t *out0);

// Adds two matrices together
void matrixSum(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0);

// Subtract two matrices
void matrixDiff(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0);

// Matrix multiplication; out0.h=in0.h, out10.w=in1.w
void matrixMultiply(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0);

// Hadamard product (Elementwise multiplication)
void hadamardProduct(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0);
// Elemetwise power of 2
void elementwisePow2(matrix32f_t *in0, matrix32f_t *out0);

// Applies ReLU to an input matrix
void relu(matrix32f_t *input0, matrix32f_t *output0);

// Complex Matrix Operations - - - - - - - - - - - - - - - - - - - - - - - - - - -
void elementwisePow2_complex(matrix32c_t *in0);
void squaredMagnitude(matrix32c_t *in0, matrix32f_t *out0);
void hadamardProduct_complex(matrix32c_t *in0, matrix32c_t *in1, matrix32c_t *out0);
void hadamardProduct_cbr(matrix32c_t *cin0, matrix32f_t *rin1, matrix32c_t *out0);

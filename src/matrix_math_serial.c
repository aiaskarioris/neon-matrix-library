#ifdef SERIAL

#include "matrix_math.h"
#include <arm_neon.h>

#ifdef DEBUG
#include <stdio.h>  // for debug messages
#endif

// Adds two matrices together
void matrixSum(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
#ifdef DEBUG
    if((in0->w != in1->w) || (in0->h != in1->h)) { printf("Error in matrixSum: (in0->w != in1->w) || (in0->h != in1->h)\n"); return; }
#endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i = 0;
    for(i; i<len; i++) { output[i] = in0->d[i] + in1->d[i]; }
}

// Subtract two matrices
void matrixDiff(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
    #ifdef DEBUG
    if((in0->w != in1->w) || (in0->h != in1->h)) { printf("Error in matrixDiff: (in0->w != in1->w) || (in0->h != in1->h)\n"); return; }
    #endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i = 0;
    for(i; i<len; i++) { output[i] = in0->d[i] + in1->d[i]; }
}

// Vector by Matrix Multiplication; if `in0.h == 0` some loops can be skipped
void multVecByMat(matrix32f_t *vec0, matrix32f_t *mat1, matrix32f_t *out0) {
#ifdef DEBUG
    // In-place multiplication isn't defined, unlike other functions
    if(out0 == NULL) { printf("Error in multVecByMat: out0==NULL\n"); return; }
    // Check vec0 is actually a vector
    if((vec0->w!=1) && (vec0->h!=1)) { printf("Error in multVecByMat: (vec0->w!=1) && (vec0->h!=1)\n"); return; }
    // Find vec0's length and check out0 is appropriately sized
    size_t vec_dim = (vec0->w > vec0->h) ? vec0->w : vec0->h;
    if((out0->h != 1) || (mat1->w != out0->w)) { printf("Error in multVecByMat: (out0->h != 1) || (mat1->w != out0->w)\n"); return; }
    if(vec_dim != mat1->h) { printf("Error in multVecByMat: vec_dim != mat1->h\n"); return; }
#endif
    size_t vec_idx;

    for(vec_idx = 0; vec_idx < mat1->w; vec_idx++) { out0->d[vec_idx] = 0.00;}

    for(size_t mat_col = 0; mat_col < mat1->w; mat_col++) {
        for(vec_idx = 0; vec_idx < mat1->h; vec_idx++) {
            out0->d[mat_col] += vec0->d[vec_idx] * mat1->d[mat_col + mat1->w*vec_idx];
        }
    }
}

// Vector by Matrix Multiplication; (when `vec0.h == 1` the vmaq optimization breaks)
void multMatByVec(matrix32f_t *mat0, matrix32f_t *vec1, matrix32f_t *out0) {
#ifdef DEBUG
    // In-place multiplication isn't defined, unlike other functions
    if(out0 == NULL) { printf("Error in multMatByVec: out0==NULL\n"); return; }
    // Check vec1 is actually a vector
    if((vec1->w!=1) && (vec1->h!=1)) { printf("Error in multMatByVec: (vec1->w!=1) && (vec1->h!=1)\n"); return; }
    // Find vec1's length and check out0 is appropriately sized
    size_t vec_dim = (vec1->w > vec1->h) ? vec1->w : vec1->h;
    if((out0->w != 1) || (mat0->h != out0->h)) { printf("Error in multMatByVec: (vec_dim != out0->w) || (mat0->h != out0->h)\n"); return; }
    if(vec_dim != mat0->w) { printf("Error in multMatByVec: mat0->w != vec_dim\n"); return; }
    // This function should be used only on multiple-of-4 vectors/matrices; Check this as well
    if((vec_dim % 4 != 0) || (mat0->w % 4 != 0)) { printf("Error in multMatByVec: (vec_dim % 4 != 0) || (mat0->w % 4 != 0)\n"); return; }
#endif
    size_t vec_idx;

    for(vec_idx = 0; vec_idx < mat0->h; vec_idx++) { out0->d[vec_idx] = 0.00;}

    for(size_t mat_row = 0; mat_row < mat0->h; mat_row++) {
        for(vec_idx = 0; vec_idx < mat0->w; vec_idx++) {
            out0->d[mat_row] += mat0->d[mat_row*mat0->w + vec_idx]*vec1->d[vec_idx];
        }
    }
}

// NOTE: Not used in algorithm
// Matrix multiplication; out0.h=in0.h, out10.w=in1.w
void matrixMultiply(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
    return;
}

// Hadamard product (Elementwise multiplication)
void hadamardProduct(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
#ifdef DEBUG
    if((in0->w != in1->w) || (in0->h != in1->h)) { printf("Error in hadamardProduct: (in0->w != in1->w) || (in0->h != in1->h)\n"); return; }
#endif

    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i = 0;
    // Handle leftovers
    for(i; i < len; i++) { output[i] = in0->d[i]*in1->d[i]; }
}

// Elemetwise power of 2
void elementwisePow2(matrix32f_t *in0, matrix32f_t *out0) {
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i = 0;

    for(i; i < len; i++) { output[i] = in0->d[i]*in0->d[i]; }
}

// Elemetwise power of 2 for complex numbers
void squaredMagnitude(matrix32c_t *in0, matrix32f_t *out0) {
    float32_t *indf = (float32_t*)in0->d;
    float32_t a,b;
    size_t len = in0->w * in0->h * 2;

    size_t o = 0;
    for(size_t i=0; i+2 < len; i+=2) {
        a = indf[i] * indf[i];
        b = indf[i+1] * indf[i+1];
        out0->d[o] = a+b;
        o++;
    }
}

// Unused function; Should be replaced by `squaredMagnitude`
void elementwisePow2_complex(matrix32c_t *in0) {
    float32_t *indf = (float32_t*)in0->d;
    size_t len = in0->w * in0->h * 2;

    for(size_t i=0; i < len; i++) { indf[i] = indf[i]*indf[i]; }
}



void relu(matrix32f_t *in0, matrix32f_t *out0) {
#ifdef DEBUG
    if(out0 != NULL) {
        if((in0->w != out0->w) || (in0->h != out0->w)) { printf("Error in relu: (in0->w != out0->w) || (in0->h != out0->w)\n"); return; }
    }
#endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i;
    for(i; i < len; i++) { in0->d[i] = (in0->d[i] < 0)? 0.0 : in0->d[i]; }
}



void hadamardProduct_complex(matrix32c_t *in0, matrix32c_t *in1, matrix32c_t *out0) {
    size_t len = in0->w * in0->h * 2;
#ifdef DEBUG
    if(in0->d == NULL || in1->d == NULL) { printf("error in hadamardProduct_complex: (in0->d == NULL || in1->d == NULL)\n"); return; }
    size_t len1 = in1->w * in1->h * 2;
    if(len != len1) { printf("Error in hadamardProduct_complex: len != len1\n"); return; }
    if(out0 != NULL) {
        if(out0->d == NULL) { printf("Error in hadamardProduct_complex: out0->d == NULL\n"); return; }
        if(out0->w != in0->w || out0->h != in0->h) { printf("Error in hadamardProduct_complex: (out0->w != in0->w || out0->h != in0->h)\n"); return; }
    }
#endif
    float32_t *indf0 = (float32_t*)in0->d;
    float32_t *indf1 = (float32_t*)in1->d;
    float32_t *outdf = (out0 != NULL) ? (float32_t*)out0->d : indf0;

    float32_t a,b,c,d;
    for(size_t i = 0; i < len; i+=2){
        a = indf0[i];
        b = indf0[i+1];
        c = indf1[i];
        d = indf1[i+1];

        outdf[i]   = a*c - b*d;
        outdf[i+1] = a*d + b*c;
    }
}

void hadamardProduct_cbr(matrix32c_t *cin0, matrix32f_t *rin1, matrix32c_t *out0) {
    size_t rlen = rin1->w * rin1->h;
    size_t clen = cin0->w * cin0->h * 2;
#ifdef DEBUG
    if(rin1->d == NULL || cin0->d == NULL) { printf("error in hadamardProduct_rbc: (rin0->d == NULL || cin1->d == NULL)\n"); return; }
    if(rlen != clen/2) { printf("Error in hadamardProduct_rbc: Mismatched input lengths\n"); return; }
    // Note: In-place multiplication is allowed
    if(out0 != NULL) {
        if(out0->d == NULL) { printf("Error in hadamardProduct_complex: out0->d == NULL\n"); return; }
        if(out0->w != cin0->w || out0->h != cin0->h) { printf("Error in hadamardProduct_rbc: (out0->w != cin0->w || cout0->h != cin0->h)\n"); return; }
    }
#endif
    float32_t *indfc = (float32_t*)cin0->d;
    float32_t *indfr = (float32_t*)rin1->d;
    float32_t *outdf = (out0 != NULL) ? (float32_t*)out0->d : indfc;

    size_t i;
    size_t r = 0; // indexes real input
    for(i = 0; i < clen; i+=2) {
        outdf[i]   = indfc[i]   * indfr[r];
        outdf[i+1] = indfc[i+1] * indfr[r];
        r++;
    }

}
#endif

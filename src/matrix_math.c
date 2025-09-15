#ifndef SERIAL

#include "matrix_math.h"
#include <arm_neon.h>

#ifdef DEBUG
#include <stdio.h>  // for debug messages
#endif

#include <stdio.h>

// extern __inline

// Adds two matrices together
void matrixSum(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL || in1->d == NULL) { printf("Error in matrixSum: (in0->d == NULL || in1->d == NULL\n"); }
    if((in0->w != in1->w) || (in0->h != in1->h)) { printf("Error in matrixSum: (in0->w != in1->w) || (in0->h != in1->h)\n"); return; }
#endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i = 0;
    float32x4_t vin0, vin1, vout0;
    while(i+4 < len) {
        vin0  = vld1q_f32(&(in0->d[i]));
        vin1  = vld1q_f32(&(in1->d[i]));
        vout0 = vaddq_f32(vin0, vin1);
        vst1q_f32(&(output[i]), vout0);
        i+=4;
    };

    for(i; i<len; i++) { output[i] = in0->d[i] + in1->d[i]; }
}

// Subtract two matrices
void matrixDiff(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL || in1->d == NULL) { printf("Error in matrixDiff: (in0->d == NULL || in1->d == NULL\n"); }
    if((in0->w != in1->w) || (in0->h != in1->h)) { printf("Error in matrixDiff: (in0->w != in1->w) || (in0->h != in1->h)\n"); return; }
#endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    size_t i = 0;
    float32x4_t vin0, vin1, vout0;
    while(i+4 < len) {
        vin0  = vld1q_f32(&(in0->d[i]));
        vin1  = vld1q_f32(&(in1->d[i]));
        vout0 = vsubq_f32(vin0, vin1);
        vst1q_f32(&(output[i]), vout0);
        i+=4;
    };

    for(i; i<len; i++) { output[i] = in0->d[i] + in1->d[i]; }
}

// Vector by Matrix Multiplication; if `in0.h == 0` some loops can be skipped
void multVecByMat(matrix32f_t *vec0, matrix32f_t *mat1, matrix32f_t *out0) {
#ifdef DEBUG
    // In-place multiplication isn't defined, unlike other functions
    if(out0 == NULL) { printf("Error in multVecByMat: out0==NULL\n"); return; }
    if(vec0->d == NULL || mat1->d == NULL || out0->d == NULL) { printf("Error in multVecByMat: (vec0->d == NULL || mat1->d == NULL || out0->d == NULL)\n"); }
    // Check vec0 is actually a vector
    if((vec0->w!=1) && (vec0->h!=1)) { printf("Error in multVecByMat: (vec0->w!=1) && (vec0->h!=1)\n"); return; }
    // Find vec0's length and check out0 is appropriately sized
    size_t vec_dim = (vec0->w > vec0->h) ? vec0->w : vec0->h;
    if((out0->h != 1) || (mat1->w != out0->w)) { printf("Error in multVecByMat: (out0->h != 1) || (mat1->w != out0->w)\n"); return; }
    if(vec_dim != mat1->h) { printf("Error in multVecByMat: vec_dim != mat1->h\n"); return; }
#endif
    // `out0` must be all zeros
    clearMatrix(out0);

    float32x4_t vin0, vrow, vout0;
    size_t mat_idx = 0;

    // Move through every element of the input vector
    for(size_t vec_idx = 0; vec_idx < mat1->h; vec_idx++) {
        vin0 = vld1q_dup_f32(&(vec0->d[vec_idx]));

        // Move through parts of each mat1 row; a row might not be a multiple of 4
        size_t pos_in_row = 0; // we need to know on which element of a row we are in
        // Move through row until less than 4 elements remain
        while(pos_in_row+4 < mat1->w){
            vrow  = vld1q_f32(&(mat1->d[mat_idx]));
            vout0 = vld1q_f32(&(out0->d[pos_in_row])); // Output is indexed by pos. in input row

            vout0 = vmlaq_f32(vout0, vin0, vrow);
            vst1q_f32(&(out0->d[pos_in_row]), vout0);

            pos_in_row += 4;
            mat_idx += 4;
        };

        // Perhaps....loading and storing back `out0` in each loop is suboptimal and it is better
        // to calculate each output vector element whole before moving to the next one. With either approach,
        // mul-and-acc is leveraged. In the alternative approach, rows are accessed first and the elements of
        // the input vector are all loaded multiple time. As a result, the question of which method is faster depends
        // (probably) on which operation is faster: vld1q_dup or vst1q_f32 (probably the former?)

        // There may be a leftover part in this row
        for(pos_in_row; pos_in_row < mat1->w; pos_in_row++) {
            out0->d[pos_in_row] += vec0->d[vec_idx]*mat1->d[mat_idx];
            mat_idx++;
        }
    }
}


// Vector by Matrix Multiplication; (when `vec0.h == 1` the vmaq optimization breaks)
void multMatByVec(matrix32f_t *mat0, matrix32f_t *vec1, matrix32f_t *out0) {
#ifdef DEBUG
    // In-place multiplication isn't defined, unlike other functions
    if(out0 == NULL) { printf("Error in multMatByVec: out0==NULL\n"); return; }
    if(mat0->d == NULL || vec1->d == NULL || out0->d == NULL) { printf("Error in multMatByVec: (mat0->d == NULL || vec1->d == NULL || out0->d == NULL)\n"); }
    // Check vec1 is actually a vector
    if((vec1->w!=1) && (vec1->h!=1)) { printf("Error in multMatByVec: (vec1->w!=1) && (vec1->h!=1)\n"); return; }
    // Find vec1's length and check out0 is appropriately sized
    size_t vec_dim = (vec1->w > vec1->h) ? vec1->w : vec1->h;
    if((out0->w != 1) || (mat0->h != out0->h)) { printf("Error in multMatByVec: (vec_dim != out0->w) || (mat0->h != out0->h)\n"); return; }
    if(vec_dim != mat0->w) { printf("Error in multMatByVec: mat0->w != vec_dim\n"); return; }
    // This function should be used only on multiple-of-4 vectors/matrices; Check this as well
    if((vec_dim % 4 != 0) || (mat0->w % 4 != 0)) { printf("Error in multMatByVec: (vec_dim % 4 != 0) || (mat0->w % 4 != 0)\n"); return; }
#endif
    // `out0` must be all zeros
    clearMatrix(out0);

    float32x4_t vvec, vmat, vtemp, vout0;

    // Move through parts of `vec1`
    for(size_t pos_in_vec = 0; pos_in_vec < mat0->w; pos_in_vec +=4) {
        vvec = vld1q_f32(&(vec1->d[pos_in_vec]));

        // Move through rows of `mat0`
        for(size_t row_idx = 0; row_idx < mat0->h; row_idx++) {
            size_t pos_in_mat = pos_in_vec + row_idx*mat0->w;
            vmat = vld1q_f32(&(mat0->d[pos_in_mat]));

            vtemp = vmulq_f32(vvec, vmat);         // multiply vector elements with matrix elements
            out0->d[row_idx] += vaddvq_f32(vtemp);   // "across-vector" addition of products and addition to output's vector element
        }
    }

    /* For the left-over vector part
    for(pos_in_vec; pos_in_vec < mat0->w; pos_in_vec++) {

    }
    */
}

// Matrix multiplication; out0.h=in0.h, out10.w=in1.w
void matrixMultiply(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
    return;
}

// Hadamard product (Elementwise multiplication)
void hadamardProduct(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL || in1->d == NULL) { printf("Error in hadamardProduct: (in0->d == NULL || in1->d == NULL\n"); }
    if((in0->w != in1->w) || (in0->h != in1->h)) { printf("Error in hadamardProduct: (in0->w != in1->w) || (in0->h != in1->h)\n"); return; }
#endif

    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    float32x4_t vin0, vin1, vout;
    size_t i = 0;

    while(i+4 < len) {
        vin0 = vld1q_f32(&(in0->d[i]));
        vin1 = vld1q_f32(&(in1->d[i]));
        vout = vmulq_f32(vin0, vin1);
        vst1q_f32(&(output[i]), vout);

        i+=4;
    }

    // Handle leftovers
    for(i; i < len; i++) { output[i] = in0->d[i]*in1->d[i]; }
}

// Elemetwise power of 2
void elementwisePow2(matrix32f_t *in0, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL) { printf("Error in elementwisePow2: in0->d==NULL\n"); return; }
#endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;
    float32x4_t vin0, vout;
    size_t i = 0;
    while(i+4 < len) {
        vin0 = vld1q_f32(&(in0->d[i]));
        vout = vmulq_f32(vin0, vin0);
        vst1q_f32(&(output[i]), vout);

        i+=4;
    }

    // Handle leftovers
    for(i; i < len; i++) { output[i] = in0->d[i]*in0->d[i]; }
}

void relu(matrix32f_t *in0, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL) { printf("Error in relu: in0->d==NULL\n"); return; }
    if(out0 != NULL) {
        if((in0->w != out0->w) || (in0->h != out0->h)) { printf("Error in relu: (in0->w != out0->w) || (in0->h != out0->w)\n"); return; }
    }
#endif
    // If `out0` is NULL store result in `in0`
    float32_t *output = (out0 == NULL) ? in0->d : out0->d;

    size_t len = in0->w * in0->h;

    float32x4_t vin, vzero;
    vzero = vld1q_dup_f32(&fzero);
    size_t i;
    for(i=0; i+4 <= len; i += 4) {
        vin = vld1q_f32(&in0->d[i]);
        vin = vmaxq_f32(vin, vzero);
        vst1q_f32(&(output[i]), vin);
    }
    for(i; i < len; i++) {
        in0->d[i] = (in0->d[i] < 0)? 0.0 : in0->d[i];
    }
}

// Complex Matrix Operations - - - - - - - - - - - - - - - - - - - - - - - - - - -
void squaredMagnitude(matrix32c_t *in0, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL) { printf("error in squaredMagnitude: in0->d==NULL\n"); return; }
    if(out0 == NULL) { printf("error in squaredMagnitude: out0==NULL\n"); return; }
    if(out0->d == NULL) { printf("error in squaredMagnitude: out0->d==NULL\n"); return; }
    if(in0->w*in0->h != out0->w*out0->h*2) { printf("error in squaredMagnitude: Mismatching Input-Output dimensions\n"); return; }
#endif
    float32_t *indf = (float32_t*)in0->d;
    float32_t *outdf = (float32_t*)out0->d;

    size_t ilen = in0->w * in0->h * 2;
    size_t olen = out0->w * out0->h;

    // `vcomplex stores 4 complex numbers`
    float32x4_t vcomplex[2], vout;

    size_t o = 0;
    size_t i;
    for(i = 0; i+8 < ilen; i+=8) {

        // Load 4 complex numbers
        vcomplex[0] = vld1q_f32(indf + i);
        vcomplex[1] = vld1q_f32(indf + i+4);
        // Calculate the squares of the components
        vcomplex[0] = vmulq_f32(vcomplex[0], vcomplex[0]);
        vcomplex[1] = vmulq_f32(vcomplex[1], vcomplex[1]);
        // Pair-wise add all components
        vout = vpaddq_f32(vcomplex[0], vcomplex[1]);
        vst1q_f32(outdf+o, vout);
        o += 4;
    }

    // Handle left-overs
    float32x2_t vreg;
    for(i; i+2 < ilen; i+=2) {
        vreg = vld1_f32(indf + i);
        vreg = vmul_f32(vreg, vreg);
        outdf[o] = vpadds_f32(vreg);
        o++;
    }
}

// Unused function; Should be replaced by `squaredMagnitude`
void elementwisePow2_complex(matrix32c_t *in0) {
#ifdef DEBUG
    if(in0->d == NULL) { printf("error in elementwisePow2_complex: in0->d==NULL\n"); return; }
#endif
    // sizeof(float complex) is 8 while sizeof(float) is 4; we'll handle in0->d as a float matrix
    float32_t *indf = (float32_t*)in0->d;

    size_t len = in0->w * in0->h * 2;
    float32x4_t vin0, vout;
    size_t i;
    for(i = 0; i+4 < len; i+=4) {
        vin0 = vld1q_f32(&(indf[i]));
        vout = vmulq_f32(vin0, vin0);
        vst1q_f32(&(indf[i]), vout);
    }

    // Handle leftovers
    for(i; i < len; i++) { indf[i] *= indf[i]; }
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
    // a+bi * c+di = ac-bd + (ad+bc)i

    // Cast input data as float*
    float32_t *indf0 = (float32_t*)(in0->d);
    float32_t *indf1 = (float32_t*)(in1->d);
    float32_t *outdf = (out0 != NULL) ? (float32_t*)(out0->d) : indf0;

    // We need registers to load 4 complex numbers from in0 and 4 from in1 (8+8 floats)
    float32x4_t vin0[2], vin1[2];
    float32x4_t vflip[2];
    float32x4_t vreal[2], vimag[2];

    // Create a SIMD reg. with 0b10..0 in all lanes; we'll use it to quickly multiply floats with -1
    const uint32_t vminus1_const[] = {0x00, 0x80000000, 0x00, 0x80000000}; // we'll load vminus with this later
    uint32x4_t vminus1 = vld1q_u32(vminus1_const);

    size_t i;
    for(i=0; i+8 <= len; i+=8) {
        vin0[0] = vld1q_f32(&(indf0[i]));
        vin0[1] = vld1q_f32(&(indf0[i+4]));

        vin1[0] = vld1q_f32(&(indf1[i]));
        vin1[1] = vld1q_f32(&(indf1[i+4]));

        // Flip vin11 and vin12 ([c1 d1 c2 d2] -> [d1 c1 d2 c2])
        vflip[0] = vrev64q_f32(vin1[0]);
        vflip[1] = vrev64q_f32(vin1[1]);

        // Do multiplications for the real parts
        vreal[0] = vmulq_f32(vin0[0], vin1[0]);
        vreal[1] = vmulq_f32(vin0[1], vin1[1]);

        // Do multiplications for the imaginary parts
        vimag[0] = vmulq_f32(vin0[0], vflip[0]);
        vimag[1] = vmulq_f32(vin0[1], vflip[1]);

        // Pairwise subtraction (real part)
        vreal[0] = vreinterpretq_f32_u32(                       // There is no pairwise subtraction instruction so we
            vorrq_u32(vminus1, vreinterpretq_u32_f32(vreal[0]))  // set the sign bits of vreal[1] to make them negative
        ); // TODO: Maybe vmulq_f32() is faster. . .
        vreal[1] = vreinterpretq_f32_u32(
            vorrq_u32(vminus1, vreinterpretq_u32_f32(vreal[1]))
        ); // TODO: Maybe vmulq_f32() is faster. . .
        vreal[0] = vpaddq_f32(vreal[0], vreal[1]);

        // Pairwise addition (imag. part)
        vimag[0] = vpaddq_f32(vimag[0], vimag[1]);

        // Store results intertwined (vreal.0 vimag.0 vreal.1 vimag.1 ...)
        vst1q_lane_f32(&(outdf[i]),   vreal[0], 0);
        vst1q_lane_f32(&(outdf[i+1]), vimag[0], 0);
        vst1q_lane_f32(&(outdf[i+2]), vreal[0], 1);
        vst1q_lane_f32(&(outdf[i+3]), vimag[0], 1);
        vst1q_lane_f32(&(outdf[i+4]), vreal[0], 2);
        vst1q_lane_f32(&(outdf[i+5]), vimag[0], 2);
        vst1q_lane_f32(&(outdf[i+6]), vreal[0], 3);
        vst1q_lane_f32(&(outdf[i+7]), vimag[0], 3);
        // _lane_ instructions need their lane argument to be const (probably to evaluate its legitimacy at compile time)
    }

    // Handle leftovers (2049 = 256*8 + 1 :[ )
    // Helper register
    vminus1 = vld1q_u32(vminus1_const); // load {0x00, 0x80, 0x00, 0x00}
    // For every pair of complex inputs 4 scalar multiplications must be made
    float32_t vbuffer0[4];
    float32_t vbuffer1[4];
    float32x4_t fm1, fm2;
    for(i; i < len; i+=2){
        // vbuffer0 = a b a b
        vbuffer0[0] = indf0[i];
        vbuffer0[1] = indf0[i+1];
        vbuffer0[2] = indf0[i];
        vbuffer0[3] = indf0[i+1];

        // vbuffer1 = c d d c
        vbuffer1[0] = indf1[i];
        vbuffer1[1] = indf1[i+1];
        vbuffer1[2] = indf1[i+1];
        vbuffer1[3] = indf1[i];

        // Do the 4 multiplications
        fm1 = vld1q_f32(vbuffer0);
        fm2 = vld1q_f32(vbuffer1);
        fm1 = vmulq_f32(fm1, fm2);

        // Mult. fm1.1 with -1 to make the subtraction
        fm1 = vreinterpretq_f32_u32(
        vorrq_u32(vminus1, vreinterpretq_u32_f32(fm1))
        ); // TODO: Maybe just vmulq_f32() is faster. . .

        // Do pairwise additions (and subtraction)
        // We'll  ignore the other two results
        fm1 = vpaddq_f32(fm1, fm1);

        // Store results
        vst1q_lane_f32(&(outdf[i]), fm1, 0);
        vst1q_lane_f32(&(outdf[i+1]), fm1, 1);
    }

    /* TODO: This part can probably be done with NEON, but is it worth it?
    float32x2_t pin0, pin1;
    float32x2_t pflip;
    float32x2_t preal, pimag;
    float32x2_t pzero = vdup_n_f32(fzero);
    for(i; i < len; i+=2) {
        pin0  = vld1_f32(&(indf0[i]));
        pin1  = vld1_f32(&(indf1[i]));
        pflip = vrev64_f32(pin1);

        preal = vmul_f32(pin0, pin1);
        pimag = vmul_f32(pin0, pflip);

        // real = preal.0 - preal.1
        // imag = pimag.0 + pimag.1
    }
    */
}

// Real Matrix x Complex Matrix
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
    // a+bi * c = ac + (bc)i

    // Cast input data as float*
    float32_t *indfc = (float32_t*)(cin0->d);
    float32_t *indfr = (float32_t*)(rin1->d);
    float32_t *outdf = (out0 != NULL) ? (float32_t*)(out0->d) : indfc;

    // We need registers to load 2 complex numbers from in0 and 2 real from in1 (4+2 floats)
    float32x4_t vcin, vrpairs;
    float32x2_t vrin[2];

    size_t i;
    size_t r = 0; // indexes real input matrix
    for(i=0; i+4 <= clen; i+=4) {
        // Load 2 complex numbers
        vcin = vld1q_f32(indfc+i);

        // Load 2 real numbers into each vrin register
        vrin[0] = vld1_dup_f32(indfr+r);
        vrin[1] = vld1_dup_f32(indfr+r+1);

        // Create a 32x4 vector with the two 32x2 vector
        vrpairs = vcombine_f32(vrin[0], vrin[1]); // = [c1 c1 c2 c2]

        // Do multiplications
        vrpairs = vmulq_f32(vcin, vrpairs);

        // Write 2 complex numbers to output (4 floats)
        vst1q_f32(outdf+i, vrpairs);

        r += 2;
    }

    // Handle leftovers (2049 = 256*8 + 1 :[ )
    // For every complex number, 2 multiplications must be done; We can use NEON for this
    float32x2_t vcin_2, vrin_2;
    for(i; i < clen; i+=2){
        vcin_2 = vld1_f32(indfc+i);
        vrin_2 = vld1_dup_f32(indfr+r);
        vcin_2 = vmul_f32(vcin_2, vrin_2);
        vst1_f32(outdf+i, vcin_2);

        r++;
    }
}
#endif


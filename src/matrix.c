#include "matrix.h"
#include <arm_neon.h>

#ifdef DEBUG
#include <stdio.h>
#endif

#include <string.h> // memset, memcpy

int newMatrix32f(size_t h, size_t w, matrix32f_t *mat) {
    float32_t *mem = (float32_t*)malloc(w*h*sizeof(float32_t));
    if(mem == NULL) { return 0; }

    mat->w  = w;
    mat->h = h;
    mat->d = mem;

    return 0;
}

int newMatrix32c(size_t h, size_t w, matrix32c_t *mat) {
    float complex *mem = (float complex*)malloc(w*h*sizeof(float complex));
    if(mem == NULL) { return 0; }

    mat->w  = w;
    mat->h = h;
    mat->d = mem;

    return 0;
}

void deleteMatrix(matrix32f_t *mat) {
    if(mat->d != NULL) {
        free(mat->d);
        mat->d = NULL;
    }
}

void clearMatrix(matrix32f_t *mat) {
    size_t len = mat->w * mat->h;
    /* Both methods take the same time
    memset(mat->d, 0, len * sizeof(float));
    return;
    */

    // Load a vector register with 0s
    float32x4_t vreg = vld1q_dup_f32(&fzero);

    // Loop until i is greater than the length of the matrix
    size_t i = 0;
    while(i+4 < len){
        vst1q_f32(&(mat->d[i]), vreg);
        i+=4;
    };

    // Zero-out leftover area (if w*h%4!=0)
    for(i; i < len; i++) { mat->d[i] = 0.00; }
}

void matrixConcat(matrix32f_t *in0, matrix32f_t *in1, matrix32f_t *out0) {
#ifdef DEBUG
    if(in0->d == NULL) { printf("Error in matrixConcat: in0 is uninitialized.\n"); return; }
    if(in1->d == NULL) { printf("Error in matrixConcat: in1 is uninitialized.\n"); return; }
    if(out0->d == NULL) { printf("Error in matrixConcat: out0 is uninitialized.\n"); return; }
#endif

    size_t len0 = in0->w  * in0->h;
    size_t len1 = in1->w  * in1->h;
    size_t leno = out0->w * out0->h;
#ifdef DEBUG
    if(len0 + len1 != leno) { printf("Error in matrixConcat: Dimensions of arguments don't match.\n"); return; }
#endif

    // Copy in0
    memcpy(out0->d, in0->d, len0 * sizeof(float32_t));
    // Copy in1
    memcpy(out0->d+len0, in1->d, len1 * sizeof(float32_t));
}


void flipVector(matrix32f_t *in0, matrix32f_t *out0) {
#ifndef SERIAL
#ifdef DEBUG
    if(in0->w != out0->w || in0->h != out0->h) { printf("Error in flipVector: (in0->w != out0->w || in0->h != out0->h)\n"); return; }
    if(in0 == out0 || in0->d == out0->d || out0 == NULL) { printf("Error in flipVector: This operation is not done in-place.\n"); return; }
    if((in0->w != 1) && (in0->h != 1)) { printf("Warning in flipVector: Attempting to flip a matrix has undefined behaviour.\n"); }
    if(in0->w * in0->h % 4) { printf("Error in flipVector: in0->w * in0->h % 4 != 0\n"); return; }
#endif

    size_t len = in0->w * in0->h;
    float32x4_t vreg;

    size_t r = len - 4;
    for(size_t i = 0; i < len; i+=4) {
        vreg = vld1q_f32(&(in0->d[i]));
        // Flip vector's contents
        vreg = vrev64q_f32(vreg); // [a b c d] => [b a d c]; that's what vrev does
        vreg = vcombine_f32(vget_high_f32(vreg), vget_low_f32(vreg));
        vst1q_f32(&(out0->d[r]), vreg);
        r -= 4;
    }
}
#else
// Serial Code * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#ifdef DEBUG
    if(in0->w != out0->w || in0->h != out0->h) { printf("Error in flipVector: (in0->w != out0->w || in0->h != out0->h)\n"); return; }
    if(in0 == out0 || in0->d == out0->d || out0 == NULL) { printf("Error in flipVector: This operation is not done in-place.\n"); return; }
    if((in0->w != 1) && (in0->h != 1)) { printf("Warning in flipVector: Attempting to flip a matrix has undefined behaviour.\n"); }
    if(in0->w * in0->h % 2) { printf("Error in flipVector: in0->w * in0->h % 2 != 0\n"); return; }
#endif

    size_t len = in0->w * in0->h;

    for(size_t i = 0; i < len; i++) {
        out0->d[len - i - 1] = in0->d[i];
    }
}
#endif // SERIAL


void dumpFloat32to8bit(matrix32f_t *mat, uint8_t int_bits, int8_t *dest) {
#ifdef DEBUG
    if(mat == NULL) { printf("Error in dumpFloat32to8bit: Input matrix is NULL\n"); return; }
    if(mat->d == NULL) { printf("Error in dumpFloat32to8bit: Input matrix is not initialized.\n"); return; }
    if(int_bits > 7) { printf("Error in dumpFloat32to8bit: `int_bits > 7`\n"); return; }
#endif

    float32_t *indf = (float32_t*)(mat->d);
    size_t len = mat->w * mat->h;
    size_t i = 0;

#ifndef SERIAL // The following code should be skipped if NEON isn't used

    // Used to index output vectors for narrowing.
    // Every 4 increments a write to memory is performed
    // and this value resets
    uint8_t step = 0;
    // Guard value that moves through 16, 12, 8 and 4 as `step` moves through
    // its values. Used to make sure that when a step cycle begins, there
    // are indeeded at least 16 numbers to be read, so that the for-loop won't
    // finish with step!=0
    size_t guard = 16;

    // Writes to `dest` happen with 16 numbers at a time; A different index is used
    size_t w_index = 0;

    // Registers for divisions
    float32x4_t vfreg, vfactor;

    // Registers for tree loading
    int16x4_t vint16x4[2]; // Once both are full, vint16x8 can be made
    int16x8_t vint16x8; // Once vint16x8 is ready, a vint8x8 can be made
    int8x8_t vint8x8[2]; // Once both are full, vint8x16 can be made
    int8x16_t vint8x16; // Once ready, a write occurs

    // Load quantization factor into vfactor
#ifdef QUANTIZE_BY_DIVISION
    vfactor = vld1q_dup_f32(quant_div_factor + (int_bits+7));
#else
    vfactor = vld1q_dup_f32(quant_mul_factor + (int_bits+7));
#endif

    for(i = 0; i+guard <= len; i+=4) {
        vfreg = vld1q_f32(indf+i);

#ifdef QUANTIZE_BY_DIVISION
        // Perform division
        vfreg = vdivq_f32(vfreg, vfactor);
#else
        // Perform multiplication
        vfreg = vmulq_f32(vfreg, vfactor);
#endif

        // Round to integer, `vqmovn` saturates the output
        vint16x4[step%2] = vqmovn_s32( // alternate between vint16[0] and vint16[1]
            vcvtnq_s32_f32(vfreg) // Round f32 to i32
        );

        // Check if the int16x4 registers should be merged (step should be 1 or 3)
        if(step%2 == 0) { // step is 0 or 2, restart loop
            step++;
            guard-=4;
            continue;
        }

        // Combine the 16x4 registers into 16x8
        vint16x8 = vcombine_s16(vint16x4[0], vint16x4[1]);

        // Write to a vint8x8 register
        // if step==1 write to 0, if step==3 write to 1
        // note: GNU compilers always divide towards 0 so this should work
#ifndef LINUX
        printf("WARNING (COMPILING WITH ELF COMPILER) ASSUMING DIVISION ROUNDS DOWN!\n\tNOT TESTED!!!\n\n");
#endif
        vint8x8[step/2] = vqmovn_s16(vint16x8);

        // Check if int8x8 registers should be merged (step should be 3)
        if(step == 3) {
            vint8x16 = vcombine_s8(vint8x8[0], vint8x8[1]);

            // Full register ready, write to memory
            vst1q_s8(dest+w_index, vint8x16);
            w_index += 16;

            // Reset step and guard
            step = 0;
            guard = 16;
        }
        // Increment step and continue
        else { step++; guard-=4; }
    }

    // NOTE:
    // The step mechanism only writes to memory once 16 numbers (or 4 quadruplets) have been read.
    // It is possible that the loop will end with `step` not 0, thus leaving processed
    // data into v-registers without being written to memory. This is why `guard` is used;
    // to stop the loop from continuing if a step cycle won't be able to complete.

    // In our application, vectors of sizes 256, 512, 1024 & 2974 will be used.
    // Only in the case of 2974 will the above scenario occur, leaving 14 numbers to be processed
    // on the serial loop.


#endif // not SERIAL

    // Handle leftovers
    float32_t ftemp;
    int8_t itemp;
    for(i; i < len; i++) {
#ifdef QUANTIZE_BY_DIVISION
        // Perform division
        ftemp = indf[i] / quant_div_factor[int_bits+7];
#else
        // Perform multiplication
        ftemp = indf[i] * quant_mul_factor[int_bits+7];
#endif
        // Perform type conversion and clamp if required
        if(ftemp < -128) // out of negative bounds
            itemp = -128;
        else if(ftemp < 128) // Withing range; no clamp
            itemp = (int8_t)ftemp;
        else // out of positive bounds
            itemp = +127;

        // Write to memory
        dest[i] = itemp;
    }
}


void matrixFrom8bit(int8_t *src, uint32_t int_bits, matrix32f_t *mat) {
#ifdef DEBUG
    if(mat == NULL) { printf("Error in matrixFrom8bit: Input matrix is NULL\n"); return; }
    if(mat->d == NULL) { printf("Error in matrixFrom8bit: Input matrix is not initialized.\n"); return; }
    if(int_bits > 7) { printf("Error in matrixFrom8bit: `int_bits > 7`\n"); return; }
#endif

    float32_t *indf = (float32_t*)(mat->d);
    size_t len = mat->w * mat->h;
    size_t i = 0;

    // Get quantization factor
    float32_t quant_factor = quant_div_factor[int_bits+7]; // This variable is used for readability

#ifndef SERIAL
    // Vector registers
    int8x8_t vreg8x8;
    int16x8_t vreg16x8;
    int32x4_t vreg32x4[2];
    float32x4_t vfreg, vfactor;

    // `dest` is read in chunks of 8 but floats are written in steps of 4;
    // A different variable is used for write indexing
    size_t out_idx = 0;

    // Load quantization factor into vector register
    vfactor = vld1q_dup_f32(&quant_factor);

    for(i; i+8 <= len; i += 8) {
        // Load 8x8 register
        vreg8x8 = vld1_s8(src + i);

        // Expand 8x8 to 16x8
        vreg16x8 = vmovl_s8(vreg8x8);

        // Expand 16x8 to 2 32x4
        vreg32x4[0] = vmovl_s16(vget_low_s16(  vreg16x8 ));
        vreg32x4[1] = vmovl_s16(vget_high_s16( vreg16x8 ));

        // Convert 32-bit integers to floats, multiply and write
        vfreg = vcvtq_f32_s32(vreg32x4[0]);
        vfreg = vmulq_f32(vfreg, vfactor);
        vst1q_f32(mat->d + out_idx, vfreg);
        out_idx += 4;

        // Repeat for second 32x4 register
        vfreg = vcvtq_f32_s32(vreg32x4[1]);
        vfreg = vmulq_f32(vfreg, vfactor);
        vst1q_f32(mat->d + out_idx, vfreg);
        out_idx += 4;

    }
#endif
    // Handle left-overs
    float32_t ftemp;
    uint32_t itemp;

    for(i; i < len; i++) {
        itemp = (int32_t)(src[i]);
        ftemp = (float32_t)itemp * quant_factor;
        mat->d[i] = ftemp;
    }
}


void matrixFrom16bit(int16_t *src, uint32_t int_bits, matrix32f_t *mat){
#ifdef DEBUG
    if(mat == NULL) { printf("Error in matrixFrom16bit: Input matrix is NULL\n"); return; }
    if(mat->d == NULL) { printf("Error in matrixFrom16bit: Input matrix is not initialized.\n"); return; }
    if(int_bits > 15) { printf("Error in matrixFrom16bit: `int_bits > 15`\n"); return; }
#endif

    float32_t *indf = (float32_t*)(mat->d);
    size_t len = mat->w * mat->h;
    size_t i = 0;

    // Get quantization factor
    float32_t quant_factor = quant_div_factor[int_bits]; // This variable is used for readability

#ifndef SERIAL
    // Vector registers
    int16x8_t vreg16x8;
    int32x4_t vreg32x4[2];
    float32x4_t vfreg, vfactor;

    // `dest` is read in chunks of 8 but floats are written in steps of 4;
    // A different variable is used for write indexing
    size_t out_idx = 0;

    // Load quantization factor into vector register
    vfactor = vld1q_dup_f32(&quant_factor);

    // Read 8-uplets of 16-bit numbers
    for(i; i+8 <= len; i += 8) {
        // Read into 6x8 register
        vreg16x8 = vld1q_s16(src + i);

        // Expand 16x8 to 2 32x4
        vreg32x4[0] = vmovl_s16(vget_low_s16(  vreg16x8 ));
        vreg32x4[1] = vmovl_s16(vget_high_s16( vreg16x8 ));

        // Convert 32-bit integers to floats, multiply and write
        vfreg = vcvtq_f32_s32(vreg32x4[0]);
        vfreg = vmulq_f32(vfreg, vfactor);
        vst1q_f32(mat->d + out_idx, vfreg);
        out_idx += 4;

        // Repeat for second 32x4 register
        vfreg = vcvtq_f32_s32(vreg32x4[1]);
        vfreg = vmulq_f32(vfreg, vfactor);
        vst1q_f32(mat->d + out_idx, vfreg);
        out_idx += 4;

    }
#endif
    // Handle left-overs
    float32_t ftemp;
    uint32_t itemp;

    for(i; i < len; i++) {
        itemp = (int32_t)(src[i]);
        ftemp = (float32_t)itemp * quant_factor;
        mat->d[i] = ftemp;
    }
}

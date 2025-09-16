#include <stdio.h> // needed for File I/O
#include <arm_neon.h>

#include "matrix.h"
#include "lut.h"

// Loads an lut32f_t object into `lut` from the file in `path`
uint8_t load32fLUT(lut32f_t *lut, const char *path) {
    uint8_t ret = 0;
    size_t bytes_read = 0;

    // Load file
    FILE *bin_file = fopen(path, "r");
    if(bin_file == NULL) { return 30; }

    // pre-allocate a 16KiB buffer to store the file
    char *buffer = malloc(sizeof(char) * 16 * 1024);
    if(buffer == NULL) { fclose(bin_file); return 100; }
    bytes_read = fread(buffer, 1, 16*1024, bin_file);
    // Exit if we couldn't even read the whole header
    if(bytes_read < (sizeof(uint32_t)* 4)) { ret = 35; goto exception; }

    // Find how many bytes long this LUT is and get the header
    uint32_t lut_length   = *(uint32_t*)(&buffer[0]);
    float32_t mult_factor = *(float32_t*)(&buffer[4]);
    float32_t bias        = *(float32_t*)(&buffer[8]);

    // Allocate memory for the LUT's content
    float32_t *data = malloc(sizeof(float32_t) * lut_length);
    if(data == NULL) { ret = 101; goto exception; }
    // Read floats; the buffer does not necessarily have all the floats
    uint32_t i;
    uint32_t buffer_idx = 4*3; // Skip the 32-bit fields in the file's header

    for(i = 0; i < lut_length; i++) {
        data[i] = *(float32_t*)(&buffer[buffer_idx]);

        buffer_idx += 4;
        // Check the buffer didn't underflow
        if(buffer_idx > 16*1024) { break; }
    }

    // Check if more floats should be read from the file
    while(i < lut_length) {
        // Read the rest of the file into our buffer
        bytes_read = fread(buffer, 1, 16*1024, bin_file);
        buffer_idx = 0;
        for(i; i < lut_length; i++) {
            data[i] = *(float32_t*)(&buffer[buffer_idx]);

            buffer_idx += 4;
            // Check the buffer didn't underflow
            if(buffer_idx > 16*1024) { break; }
        }
    }
    fclose(bin_file);
    free(buffer);

    // Put data into `lut` input argument
    lut->length         = lut_length;
    lut->mult_factor    = mult_factor;
    lut->bias           = bias;
    lut->data = data;

    return 0;

exception:
    fclose(bin_file);
    free(buffer);
    return ret;
}

void deleteLUT32f(lut32f_t *lut) {
    if(lut->data != NULL) {
        free(lut->data);
        lut->data = NULL;
    }
}

#ifndef SERIAL
// NEON Code * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
void clampingLUT(matrix32f_t *input0, lut32f_t *lut, matrix32f_t *output0) {
#ifdef DEBUG
    if(input0->d == NULL) { printf("Error in clampingLUT: input0 is not initiated.\n"); return; }
    if(output0 != NULL) { // In-place LUT passing is possible but the following test has no meaning in that case
        if(output0->d == NULL) { printf("Error in clampingLUT: output0 is not initiated.\n"); return; }
        if((input0->w != output0->w) || (input0->h != output0->h)) {
            printf("Error in clampingLUT: (input0.w != output0.w) || (input0.h != output0.h)\n");
            return;
        }
    if(lut->data == NULL) { printf("Error in clampingLUT: The LUT is not initiated.\n"); return; }

#endif
    // Note a matrix' dimensions have no effect when applying an LUT.
    size_t length = input0->h * input0->w;
    // Select the active output
    float32_t *output = (output0 == NULL) ? input0->d : output0->d;

    // Multiplication factor for normalizing input to lut length
    float32_t mult_factor = lut->mult_factor;
    // Bias for mapping negative inputs to positive values
    float32_t bias = lut->bias; // (= half of the LUT's length, without counting the output for 0)

    // Lovely registers
    float32x4_t vfin;
    uint32x4_t vuint;
    // Register for multiplying inputs with the multiplication factor
    float32x4_t vfactor = vld1q_dup_f32(&mult_factor);
    // Register for applying bias, getting negative values to positive range
    float32x4_t vbias = vld1q_dup_f32(&bias);
    // Vector filled with 0.00s, used to trim negative numbers after normalization
    float32x4_t vzero = vld1q_dup_f32(&fzero);
    // Like the above but for clamping positive numbers
    uint32_t last_lut_idx = lut->length - 1;
    uint32x4_t  vlutlen = vld1q_dup_u32(&last_lut_idx);

    // Use SIMD
    size_t i;
    for(i = 0; i+4 <= length; i+= 4) {
        vfin = vld1q_f32(&input0->d[i]);
        // Normalize and represent as multiple of minimum and apply bias
        vfin = vmlaq_f32(vbias, vfactor, vfin);

        /*// This may or may not be faster
        vfin = vmulq_n_f32(vfin, mult_factor);
        vfin = vaddq(vfin, vbias);
        */

        // Trim negative numbers to 0
        vfin = vmaxq_f32(vfin, vzero);

        // Convert input vector to integer vector
        vuint = vcvtnq_u32_f32(vfin);

        // Trim inputs greater than the length of the LUT
        vuint = vminq_u32(vuint, vlutlen);

        // At this point the contents of vint can be used to replace input0->d[i..i+3]
        output[i+0] = lut->data[ vgetq_lane_u32(vuint, 0) ];
        output[i+1] = lut->data[ vgetq_lane_u32(vuint, 1) ];
        output[i+2] = lut->data[ vgetq_lane_u32(vuint, 2) ];
        output[i+3] = lut->data[ vgetq_lane_u32(vuint, 3) ];
    }

    // Get leftover numbers
    float32_t ftemp;
    uint32_t  utemp;
    for(i; i < length; i++) {
        ftemp = input0->d[i] * mult_factor + bias;

        // Clamp negative numbers
        ftemp = (ftemp < 0.0)? 0.0 : ftemp;

        // Convert to uint32 and clamp big numbers
        utemp = (uint32_t)ftemp;
        utemp = (utemp > lut->length)? lut->length : utemp;

        output[i] = lut->data[utemp];
    }
}

void sqrtLUT(matrix32f_t *input0, lut32f_t *lut, matrix32f_t *output0) {
#ifdef DEBUG
    if(input0->d == NULL) { printf("Error in sqrtLUT: input0 is not initialized.\n"); return; }
    if(output0 != NULL) { // In-place LUT passing is possible but the following tests have no meaning in that case
        if(output0->d == NULL) { printf("Error in sqrtLUT: output0 is not initialized.\n"); return; }
        if((input0->w != output0->w) || (input0->h != output0->h)) {
            printf("Error in sqrtLUT: (input0.w != output0.w) || (input0.h != output0.h)\n");
            return;
        }
        if(lut->data == NULL) { printf("Error in sqrtLUT: The LUT is not initiliazed.\n"); return; }
    }
#endif
    // NOTE: A matrix' dimensions have no effect when applying an LUT.
    size_t length = input0->h * input0->w;
    // Select the active output
    float32_t *output = (output0 == NULL) ? input0->d : output0->d;

    // Multiplication factor for normalizing input to lut length
    const float32_t mult_factor = lut->mult_factor;

    // Use SIMD
    float32x4_t vfin;
    uint32x4_t vuint;
    size_t i;
    for(i = 0; i+4 <= length; i+= 4) {
        vfin = vld1q_f32(&input0->d[i]);
        // Normalize and represent as multiple of minimum with one instruction
        vfin = vmulq_n_f32(vfin, mult_factor);

#ifdef DEBUG
        if(vmaxvq_f32(vfin) > lut->length) {
            printf("sqrt_lut(%x[%d], %x): A number in this input exceeds the pre-defined maximum.\n", input0, i, output0);
        }
#endif

        // Convert input vector to integer vector
        vuint = vcvtnq_u32_f32(vfin);

        // At this point the contents of vint can be used to replace input0->d[i..i+3]
        output[i+0] = lut->data[ vgetq_lane_u32(vuint, 0) ];
        output[i+1] = lut->data[ vgetq_lane_u32(vuint, 1) ];
        output[i+2] = lut->data[ vgetq_lane_u32(vuint, 2) ];
        output[i+3] = lut->data[ vgetq_lane_u32(vuint, 3) ];
    }

    // Get leftover numbers
    float32_t ftemp;
    uint32_t  utemp;
    for(i; i < length; i++) {
        ftemp = input0->d[i] * mult_factor;

#ifdef DEBUG
        if(ftemp > lut->length) {
            printf("sqrt_lut(%x[%d], %x): A number in this input exceeds the pre-defined maximum.\n", input0, i, output0);
        }
#endif
        utemp = (uint32_t)ftemp;
        output[i] = lut->data[utemp];
    }
}

void angleLUT_c(matrix32c_t *input0, lut32f_t *lut, matrix32f_t *output0) {
#ifdef DEBUG
    if(input0->d == NULL || output0->d == NULL) { printf("Error in angleLUT: Input/Output not initialized.\n"); return; }
    if(lut==NULL) { printf("Error in angleLUT: LUT==NULL.\n"); return; }
#endif
    size_t len = input0->w * input0->h * 2;

    // Cast input/output data pointers as convinient floats
    float32_t *indf  = (float32_t*)input0->d;
    float32_t *outdf = (float32_t*)output0->d;

    float32_t fbuffer[4];
    float32x4_t vreal, vimag, vdiv;
    uint32x4_t vuint;

    // Variables/registers used for float to int conversion (see `clampintLUT()`)
    float32_t mult_factor = lut->mult_factor;
    float32_t bias = lut->bias;
    uint32_t last_lut_idx = lut->length - 1;

    float32x4_t vfactor = vld1q_dup_f32(&mult_factor);
    float32x4_t vbias   = vld1q_dup_f32(&bias);
    float32x4_t vzero   = vld1q_dup_f32(&fzero);
    uint32x4_t  vlutlen = vld1q_dup_u32(&last_lut_idx);

    // i indexes the complex input, o indexes the real output
    size_t i;
    size_t o = 0;
    for(i = 0; i+8 <= len; i+=8) {
        // There is no pairwise division Neon instruction so
        // each float is loaded to its respective vregisters manually
        for(uint8_t t = 0; t < 4; t++){ fbuffer[t] = indf[i+ t*2]; }
        vreal = vld1q_f32(fbuffer);

        for(uint8_t t = 0; t < 4; t++){ fbuffer[t] = indf[i+1+ t*2]; }
        vimag = vld1q_f32(fbuffer);

        vdiv = vdivq_f32(vimag, vreal);

        // Continue with the usual LUT operation...
        // Normalize and represent as multiples of minimum
        vdiv = vmlaq_f32(vbias, vfactor, vdiv);

        // Trim negative numbers to zero
        vdiv = vmaxq_f32(vdiv, vzero);

        // Convert input vector to integer vector
        vuint = vcvtnq_u32_f32(vdiv);

        // Trim inputs greater than the length of the LUT
        vuint = vminq_u32(vuint, vlutlen);

        // At this point the contents of vint can be used to replace input0->d[i..i+3]
        outdf[o+0] = lut->data[ vgetq_lane_u32(vuint, 0) ];
        outdf[o+1] = lut->data[ vgetq_lane_u32(vuint, 1) ];
        outdf[o+2] = lut->data[ vgetq_lane_u32(vuint, 2) ];
        outdf[o+3] = lut->data[ vgetq_lane_u32(vuint, 3) ];

        o += 4;
    }

    // Handle leftovers
    float32_t ftemp;
    uint32_t  utemp;
    for(i; i < len; i+=2) {
        ftemp = indf[i+1] / indf[i];
        ftemp = ftemp * mult_factor + bias;

        ftemp = (ftemp < 0.0)? 0.0 : ftemp;

        utemp = (uint32_t)ftemp;
        utemp = (utemp > lut->length)? lut->length : utemp;
        outdf[o] = lut->data[utemp];
        o++;
    }
}

void expiLUT(matrix32f_t *input0, lut32f_t *sinlut, lut32f_t *coslut, matrix32c_t *output0) {
#ifdef DEBUG
    if(input0 == NULL) { printf("Error in expiLUT: in-place operation is not supported.\n"); return; }
    if(input0->d == NULL) { printf("Error in expiLUT: input0 is not initialized.\n"); return; }
    if(output0->d == NULL) { printf("Error in expiLUT: output0 is not initialized.\n"); return; }
    if(sinlut->data == NULL) { printf("Error in expiLUT: Sine LUT is not initialized.\n"); return; }
    if(coslut->data == NULL) { printf("Error in expiLUT: Cosine LUT is not initialized.\n"); return; }
    if(sinlut->length != coslut->length) { printf("Error in expiLUT: Sine and Cosine LUTs should have the same lengths.\n"); return; }
#endif
    // NOTE: A matrix' dimensions have no effect when applying an LUT.
    size_t len = input0->h * input0->w;

    // Cast input/output data pointers as convinient floats
    float32_t *indf  = (float32_t*)input0->d;
    float32_t *outdf = (float32_t*)output0->d;

    float32x4_t vfin;
    uint32x4_t  vuint;

    float32_t mult_factor = sinlut->mult_factor;
    float32_t bias = sinlut->bias;

    float32x4_t vfactor = vld1q_dup_f32(&mult_factor);
    float32x4_t vbias   = vld1q_dup_f32(&bias);

    // It should be possible to only use one LUT for this operation, with sine([-pi2, pi]).
    // This method would use 3/4 of memory used in 2 LUTs, since now we have sin([0, p/2]) stored
    // twice, once in each lut. However, this method requires more tinkering with the LUT mechanism.

    // Use SIMD
    size_t i;
    size_t o = 0;
    for(i = 0; i+4 <= len; i+= 4) {
        vfin = vld1q_f32(indf+i);

        // Normalize and represent as multiple of minimum
        vfin = vmlaq_f32(vbias, vfactor, vfin);

        // Convert input vector to integer vector
        vuint = vcvtnq_u32_f32(vfin);

        // This function's input is always the output of `angleLUT_c`, which is always within the range [-pi/2, +pi/2]
        // There is no need to clamp or check for problematic input values

        // Find sine(n) for imaginery part
        outdf[o+1] = sinlut->data[ vgetq_lane_u32(vuint, 0) ];
        outdf[o+3] = sinlut->data[ vgetq_lane_u32(vuint, 1) ];
        outdf[o+5] = sinlut->data[ vgetq_lane_u32(vuint, 2) ];
        outdf[o+7] = sinlut->data[ vgetq_lane_u32(vuint, 3) ];

        // Offset vuint to find cosine(n) for real part
        outdf[o] = coslut->data[ vgetq_lane_u32(vuint, 0) ];
        outdf[o+2] = coslut->data[ vgetq_lane_u32(vuint, 1) ];
        outdf[o+4] = coslut->data[ vgetq_lane_u32(vuint, 2) ];
        outdf[o+6] = coslut->data[ vgetq_lane_u32(vuint, 3) ];

        o += 8;
    }

    // Get leftover numbers
    float32_t ftemp;
    uint32_t  utemp;
    for(i; i < len; i++) {
        ftemp = indf[i];
        ftemp = ftemp * mult_factor + bias;

        // Get sine
        utemp = (uint32_t)ftemp;
        outdf[o+1] = sinlut->data[utemp];

        // Get cosine
        outdf[o] = coslut->data[utemp];

        o+=2;
    }
}

#else
// Serial Code * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
void clampingLUT(matrix32f_t *input0, lut32f_t *lut, matrix32f_t *output0) {
#ifdef DEBUG
    if(input0->d == NULL) { printf("Error in clampingLUT: input0 is not initiated.\n"); return; }
    if(output0 != NULL) { // In-place LUT passing is possible but the following test has no meaning in that case
        if(output0->d == NULL) { printf("Error in clampingLUT: output0 is not initiated.\n"); return; }
        if((input0->w != output0->w) || (input0->h != output0->h)) {
            printf("Error in clampingLUT: (input0.w != output0.w) || (input0.h != output0.h)\n");
            return;
        }
        if(lut->data == NULL) { printf("Error in clampingLUT: The LUT is not initiated.\n"); return; }
    }
#endif
    // Note a matrix' dimensions have no effect when applying an LUT.
    size_t length = input0->h * input0->w;
    // Select the active output
    float32_t *output = (output0 == NULL) ? input0->d : output0->d;

    // Multiplication factor for normalizing input to lut length
    float32_t mult_factor = lut->mult_factor;
    // Bias for mapping negative inputs to positive values
    float32_t bias = lut->bias; // (= half of the LUT's length, without counting the output for 0)
    // Like the above but for clamping positive numbers
    uint32_t last_lut_idx = lut->length - 1;

    // Get leftover numbers
    float32_t ftemp;
    uint32_t  utemp;
    for(size_t i; i < length; i++) {
        ftemp = input0->d[i] * mult_factor + bias;

        // Clamp negative numbers
        ftemp = (ftemp < 0.0)? 0.0 : ftemp;

        // Convert to uint32 and clamp big numbers
        utemp = (uint32_t)ftemp;
        utemp = (utemp > lut->length)? lut->length : utemp;

        output[i] = lut->data[utemp];
    }
}

void sqrtLUT(matrix32f_t *input0, lut32f_t *lut, matrix32f_t *output0) {
    #ifdef DEBUG
    if(input0->d == NULL) { printf("Error in sqrtLUT: input0 is not initiated.\n"); return; }
    if(output0 != NULL) { // In-place LUT passing is possible but the following test has no meaning in that case
        if(output0->d == NULL) { printf("Error in sqrtLUT: output0 is not initiated.\n"); return; }
        if((input0->w != output0->w) || (input0->h != output0->h)) {
            printf("Error in sqrtLUT: (input0.w != output0.w) || (input0.h != output0.h)\n");
            return;
        }
        if(lut->data == NULL) { printf("Error in sqrtLUT: The LUT is not initiated.\n"); return; }
    }
    #endif
    // NOTE: A matrix' dimensions have no effect when applying an LUT.
    size_t length = input0->h * input0->w;
    // Select the active output
    float32_t *output = (output0 == NULL) ? input0->d : output0->d;

    // Multiplication factor for normalizing input to lut length
    const float32_t mult_factor = lut->mult_factor;

    // Get leftover numbers
    float32_t ftemp;
    uint32_t  utemp;
    for(size_t i = 0; i < length; i++) {
        ftemp = input0->d[i] * mult_factor;

        #ifdef DEBUG
        if(ftemp > lut->length) {
            printf("sqrt_lut(%x[%d], %x): A number in this input exceeds the pre-defined maximum.\n", input0, i, output0);
        }
        #endif
        utemp = (uint32_t)ftemp;
        output[i] = lut->data[utemp];
    }
}

void angleLUT_c(matrix32c_t *input0, lut32f_t *lut, matrix32f_t *output0) {
#ifdef DEBUG
    if(input0->d == NULL || output0->d == NULL) { printf("Error in angleLUT: Input/Output not initialized.\n"); return; }
    if(lut==NULL) { printf("Error in angleLUT: LUT==NULL.\n"); return; }
#endif
    size_t len = input0->w * input0->h * 2;

    // Multiplication factor for normalizing input to lut length
    float32_t mult_factor = lut->mult_factor;
    // Bias for mapping negative inputs to positive values
    float32_t bias = lut->bias; // (= half of the LUT's length, without counting the output for 0)
    // Like the above but for clamping positive numbers
    uint32_t last_lut_idx = lut->length - 1;

    float32_t *indf  = (float32_t*)input0->d;
    float32_t *outdf = (float32_t*)output0->d;

    // Handle leftovers
    float32_t ftemp;
    uint32_t  utemp;
    size_t o = 0; // indexes real output
    for(size_t i; i < len; i+=2) {
        ftemp = indf[i+1] / indf[i];
        ftemp = ftemp * mult_factor + bias;

        ftemp = (ftemp < 0.0)? 0.0 : ftemp;

        utemp = (uint32_t)ftemp;
        utemp = (utemp > lut->length)? lut->length : utemp;
        outdf[o] = lut->data[utemp];
        o++;
    }
}

void expiLUT(matrix32f_t *input0, lut32f_t *sinlut, lut32f_t *coslut, matrix32c_t *output0) {
#ifdef DEBUG
    if(input0 == NULL) { printf("Error in expiLUT: in-place operation is not supported.\n"); return; }
    if(input0->d == NULL) { printf("Error in expiLUT: input0 is not initialized.\n"); return; }
    if(output0->d == NULL) { printf("Error in expiLUT: output0 is not initialized.\n"); return; }
    if(sinlut->data == NULL) { printf("Error in expiLUT: Sine LUT is not initialized.\n"); return; }
    if(coslut->data == NULL) { printf("Error in expiLUT: Cosine LUT is not initialized.\n"); return; }
    if(sinlut->length != coslut->length) { printf("Error in expiLUT: Sine and Cosine LUTs should have the same lengths.\n"); return; }
#endif
    size_t len = input0->w * input0->h;

    // Multiplication factor for normalizing input to lut length
    float32_t mult_factor = sinlut->mult_factor;
    // Bias for mapping negative inputs to positive values
    float32_t bias = sinlut->bias; // (= half of the LUT's length, without counting the output for 0)
    // Like the above but for clamping positive numbers
    uint32_t last_lut_idx = sinlut->length - 1;

    float32_t *indf  = input0->d;
    float32_t *outdc = (float32_t*)output0->d;

    // Get leftover numbers
    float32_t ftemp;
    uint32_t  utemp;
    size_t o = 0; // used for indexing complex output
    for(size_t i = 0; i < len; i++) {
        ftemp = indf[i];
        ftemp = ftemp * mult_factor + bias;

        // Get sine
        utemp = (uint32_t)ftemp;
        outdc[o+1] = sinlut->data[utemp];

        // Get cosine
        outdc[o] = coslut->data[utemp];

        o+=2;
    }
}

#endif


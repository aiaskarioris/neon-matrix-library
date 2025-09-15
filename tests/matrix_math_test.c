#include <stdio.h>
#include <string.h>

#include "csv.h"
#include "matrix_math.h"
#include "lut.h"
#include "stft.h"

#include "clock.h"

#include "functions.h"

float32_t f32abs(float32_t f) { return (f>=0)?f:(-1.0*f); }

int main(int argc, char **argv) {
	uint8_t ret = 0;
	printf("Aias Karioris, 2025\n");
	printf("Matrix Math Routines Tests");
#ifndef SERIAL
	printf(" (NEON)");
#endif
#ifdef DEBUG
	printf(" [Debug Build]");
#endif
	printf("\n\n");

	lut32f_t lut0; lut0.data = NULL;
	lut32f_t lut1; lut1.data = NULL;
	matrix32f_t input1, input2, output1, expected_output;
	input1.d = NULL, input2.d = NULL, output1.d = NULL, expected_output.d = NULL;
	matrix32c_t cinput1, cinput2, coutput1, cexpected_output;
	cinput1.d = NULL; cinput2.d = NULL; coutput1.d = NULL; cexpected_output.d = NULL;

	if(argc!=9 && argc!=6) {
		printf("Usage: %s [function] [h1] [w1] [input1] [h2] [w2] [input2] [expected-output]\nFunctions can be: ", argv[0]);
		for(uint8_t f = 0; f < valid_function_count; f++) { printf("%s ", valid_functions_str[f]); }
		printf("\n\n");
		return 1;
	}

	// Check function is valid
	function_t selected_function = 0;
	while(strcmp(valid_functions_str[selected_function], argv[1])) {
		selected_function++;
		if(selected_function == valid_function_count) { break; }
	}
	if(selected_function == valid_function_count) {
		printf("Error: Function %s is invalid.\nPossible options are:\n\t", argv[1]);
		for(uint8_t i = 0; i < valid_function_count; i++) { printf("%s ", valid_functions_str[i]); }
		ret = 2; goto exit;
	}
	else { printf("\nSelected Function: %s (#%d)\n", valid_functions_str[selected_function], (uint32_t)selected_function); }

	// Get dimensions from file names
	// The following size refer to the number of elements (not floats as in 2 floats per complex element)
	size_t w1, h1, w2=0, h2=0, wo, ho;
	h1 = atoi(argv[2]);	w1 = atoi(argv[3]);
	if(selected_function <= hadamardProductEnum || selected_function == hadamardProduct_complexEnum || selected_function == hadamardProduct_cbrEnum) { // Some functions require 2 inputs
		h2 = atoi(argv[5]); w2 = atoi(argv[6]);
	}


	// Configure input/output data dimensions and load LUTs
	switch(selected_function) {
		case matrixSumEnum:
			wo = w1; ho = h1; break;
		case matrixDiffEnum:
			wo = w1; ho = h1; break;
		case multVecByMatEnum:
			ho = 1; wo = w2; break;
		case multMatByVecEnum:
			ho = h1; wo = 1; break;
		case hadamardProductEnum:
			wo = w1; ho = h1; break;
		case elementwisePow2Enum:
			wo = w1; ho = h1; break;
		case reluEnum:
			wo = w1; ho = h1; break;
		case sqrtLutEnum:
			if(load32fLUT(&lut0, "lut/sqrt256.lut")) {
				printf("Error: Could not load Square Root LUT.\n\n");
				ret = 10; goto exit;
			};
			wo = w1; ho = h1; break;
		case tanhLutEnum:
			if(load32fLUT(&lut0, "lut/tanh65537.lut")) {
				printf("Error: Could not load Tanh LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case sigmoidLutEnum:
			if(load32fLUT(&lut0, "lut/sigmoid65537.lut")) {
				printf("Error: Could not load Sigmoid LUT.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case flipEnum:
			wo = w1; ho = h1; break;
		case extend2Enum:
			wo = w1*2; ho = h1; break;
		case extend4Enum:
			wo = w1*4; ho = h1; break;
		case extend8Enum:
			wo = w1*8; ho = h1; break;
		case squaredMagnitudeEnum:
			wo = w1; ho = h1; break;
		case hadamardProduct_complexEnum:
			wo = w1; ho = h1; break;
		case angleLutEnum:
			if(load32fLUT(&lut0, "lut/atan65537.lut")) {
				printf("Error: Could not load atan LUT (angle).\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case expiLutEnum:
			if(load32fLUT(&lut0, "lut/sin256.lut") || load32fLUT(&lut1, "lut/cos256.lut")) {
				printf("Error: Could not load sine/cosine LUTs.\n\n");
				ret = 10; goto exit;
			}
			wo = w1; ho = h1; break;
		case hadamardProduct_cbrEnum:
			wo = w1; ho = h1; break;
		default:
			ret = -2; goto exit;
	}

	// Load input(s) and configure output
	uint8_t test = 0;
	startClock();

	// Expect 2 real inputs
	if(selected_function <= hadamardProductEnum) {
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		startClock();
		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h2, w2, &input2)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[7]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected output
		startClock();
		printf("Loading %s...", argv[8]);
		if(test = matrixFromCSV(argv[8], ho, wo, &expected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[8]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for our output
		if(newMatrix32f(ho, wo, &output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
	}
	// Expect 1 (real) input
	else if(selected_function <= extend8Enum){
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected output
		startClock();
		printf("Loading %s...", argv[5]);
		if(test = matrixFromCSV(argv[5], ho, wo, &expected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[5]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for our output
		if(newMatrix32f(ho, wo, &output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
	}
	// Expect 2 complex inputs
	else if(selected_function == hadamardProduct_complexEnum){
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		startClock();
		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h2, w2*2, (matrix32f_t*)&cinput2)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[7]);
			ret = 3; goto exit;
		}
		cinput2.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected complex output
		startClock();
		printf("Loading %s...", argv[8]);
		if(test = matrixFromCSV(argv[8], ho, wo*2, (matrix32f_t*)&cexpected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[8]);
			ret = 3; goto exit;
		}
		cexpected_output.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&coutput1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
	}
	// Expect 1 complex input and real output
	else if(selected_function == squaredMagnitudeEnum){
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected output
		startClock();
		printf("Loading %s...", argv[5]);
		if(test = matrixFromCSV(argv[5], ho, wo, &expected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[5]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create complex matrix for our output
		if(newMatrix32f(ho, wo, &output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
	}
	// Expect complex input, config. real output
	else if(selected_function == angleLutEnum) {
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected output
		startClock();
		printf("Loading %s...", argv[5]);
		if(test = matrixFromCSV(argv[5], ho, wo, &expected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[5]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create matrix for our output
		if(newMatrix32f(ho, wo, (matrix32f_t*)&output1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
	}
	// Expect real input, config. complex output
	else if(selected_function == expiLutEnum) {
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected output
		startClock();
		printf("Loading %s...", argv[5]);
		if(test = matrixFromCSV(argv[5], ho, wo*2, (matrix32f_t*)&cexpected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[5]);
			ret = 3; goto exit;
		}
		cexpected_output.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&coutput1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
	}
	// Expect one real and one complex input, config. complex output
	else if(selected_function == hadamardProduct_cbrEnum) {
		// Complex input (first)
		printf("Loading %s...", argv[4]);
		if(test = matrixFromCSV(argv[4], h1, w1*2, (matrix32f_t*)&cinput1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		cinput1.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Real input (second)
		printf("Loading %s...", argv[7]);
		if(test = matrixFromCSV(argv[7], h1, w1, &input1)) {
			printf("\nError (%d): failed to import %s!\n\n", test, argv[4]);
			ret = 3; goto exit;
		}
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Load expected output
		startClock();
		printf("Loading %s...", argv[8]);
		if(test = matrixFromCSV(argv[8], ho, wo*2, (matrix32f_t*)&cexpected_output)) {
			printf("Error (%d): failed to import %s!\n\n", test, argv[5]);
			ret = 3; goto exit;
		}
		cexpected_output.w /= 2;
		printf("OK! (%.2f ms)\n", clockToMS(readClock()));

		// Create complex matrix for our output
		if(newMatrix32f(ho, wo*2, (matrix32f_t*)&coutput1)) {
			printf("Error: failed to create an output matrix.\n\n");
			ret = -3; goto exit;
		}
		coutput1.w /= 2;
	}

	// Perform test
	switch(selected_function) {
		case matrixSumEnum:
			startClock(); matrixSum(&input1, &input2, &output1); break;
		case matrixDiffEnum:
			startClock(); matrixSum(&input1, &input2, &output1); break;
		case multVecByMatEnum:
			startClock(); multVecByMat(&input1, &input2, &output1);	break;
		case multMatByVecEnum:
			startClock(); multMatByVec(&input1, &input2, &output1); break;
		case hadamardProductEnum:
			startClock(); hadamardProduct(&input1, &input2, &output1); break;
		case elementwisePow2Enum:
			startClock(); elementwisePow2(&input1, &output1); break;
		case reluEnum:
			startClock(); relu(&input1, &output1); break;
		case sqrtLutEnum:
			startClock(); sqrtLUT(&input1, &lut0, &output1); break;
		case tanhLutEnum:
			startClock(); clampingLUT(&input1, &lut0, &output1); break;
		case sigmoidLutEnum:
			startClock(); clampingLUT(&input1, &lut0, &output1); break;
		case flipEnum:
			startClock(); flipVector(&input1, &output1); break;
		case extend2Enum:
			startClock(); extendInput(&input1, &output1, 2); break;
		case extend4Enum:
			startClock(); extendInput(&input1, &output1, 4); break;
		case extend8Enum:
			startClock(); extendInput(&input1, &output1, 8); break;
		case squaredMagnitudeEnum:
			startClock(); squaredMagnitude(&cinput1, &output1); break;
			// note: Complex matrix math functions are hard-coded to have no output arg.
		case hadamardProduct_complexEnum:
			startClock(); hadamardProduct_complex(&cinput1, &cinput2, &coutput1); break;
		case angleLutEnum:
			startClock(); angleLUT_c(&cinput1, &lut0, &output1); break;
		case expiLutEnum:
			startClock(); expiLUT(&input1, &lut0, &lut1, &coutput1); break;
		case hadamardProduct_cbrEnum:
			startClock(); hadamardProduct_cbr(&cinput1, &input1, &coutput1); break;
		default:
			ret = -2; goto exit;
	}
	printf("Test completed: %.3fms\n", clockToMS(readClock()));

	// Display results if they aren't too many
	if(cinput1.d) {
		if(cinput1.w*cinput1.h <= 16*32) {
			printf("\nInput1:");
			for(size_t i = 0 ; i < cinput1.w*cinput1.h*2; i+=2) { printf("%c%+03.2f%+02.3fi", (i%16)?' ':'\n', ((float32_t*)cinput1.d)[i], ((float32_t*)cinput1.d)[i+1]); }
		}
		if(cinput2.d != NULL && cinput2.w*cinput2.h <= 16*32) {
			printf("\nInput2:");
			for(size_t i = 0 ; i < cinput2.w*cinput2.h*2; i+=2) { printf("%c%+03.2f%+02.3fi", (i%16)?' ':'\n', ((float32_t*)cinput2.d)[i], ((float32_t*)cinput2.d)[i+1]); }
		}
	}
	if(input1.d) {
		if(input1.w*input1.h <= 16*32) {
			printf("\nInput1:");
			for(size_t i = 0 ; i < input1.w*input1.h; i++) { printf("%c%+05.3f", (i%32)?' ':'\n', input1.d[i]); }
		}
		if(input2.d != NULL && input2.w*input2.h <= 16*32) {
			printf("\nInput2:");
			for(size_t i = 0 ; i < input2.w*input2.h; i++) { printf("%c%+05.3f", (i%32)?' ':'\n', input2.d[i]); }
		}
	}

	if(output1.d) {
		printf("\nOutput:");
		for(size_t i = 0 ; i < output1.w*output1.h; i++) { printf("%c%+05.3f", (i%32)?' ':'\n', output1.d[i]); }
	}
	else if(coutput1.d) {
		printf("\nOutput:");
		for(size_t i = 0 ; i < coutput1.w*coutput1.h*2; i+=2) { printf("%c%+03.2f%+02.2fi", (i%16)?' ':'\n', ((float32_t*)coutput1.d)[i]*1000.0, ((float32_t*)coutput1.d)[i+1]*1000.0); }
	}
	printf("\n");

	// Check if the results match in dimensions
	if(output1.d){
		if((expected_output.h != output1.h) || (expected_output.w != output1.w)) {
			printf("Fail: Output matrices don't match in dimensions!\nExpected %dx%d, got %dx%d\n\n", expected_output.h,expected_output.w, output1.h,  output1.w);
			ret = 10; goto exit;
		}
	}
	else if(coutput1.d) {
		if((cexpected_output.h != coutput1.h) || (cexpected_output.w != coutput1.w)) {
			printf("Fail: Output matrices don't match in dimensions!\nExpected %dx%d, got %dx%d\n\n", cexpected_output.h,cexpected_output.w, coutput1.h, coutput1.w);
			ret = 10; goto exit;
		}
	}


	// Check if the expected matrix matches our result
	float32_t *expfd = (expected_output.d != NULL) ? expected_output.d : (float32_t*)cexpected_output.d;
	float32_t *outfd  = (output1.d != NULL) ? output1.d : (float32_t*)coutput1.d;
	size_t outlen = (expected_output.d != NULL) ? expected_output.w*expected_output.h : cexpected_output.w*cexpected_output.h*2;

	float32_t err = 0.0;
	float32x4_t vexp, vout;
	size_t idx = 0;
	while(idx+4 < outlen) {
		vexp = vld1q_f32(&(expfd[idx]));
		vout = vld1q_f32(&(outfd[idx]));
		vout = vsubq_f32(vexp, vout);
		err  += f32abs(vaddvq_f32(vout));
		//printf("\n\t\t\t\t\t\t\t%2.6f += %2.6f\r", err, f32abs(vaddvq_f32(vout)));
		idx+=4;
	}
	for(idx; idx < outlen; idx++) {
		err += f32abs(expfd[idx] - outfd[idx]);
		//printf("\n\t\t\t\t\t\t\t%2.6f += %2.6f\r", err, expected_output.d[idx] - output1.d[idx]);
	}
	err /= (float32_t)outlen;

	printf("Done testing! Mean Error between results: %3.4f\n", err);
	printf("\n");
exit:
	deleteMatrix(&input1);
	deleteMatrix(&input2);
	deleteMatrix(&output1);
	deleteMatrix(&expected_output);

	deleteMatrix((matrix32f_t*)&cinput1);
	deleteMatrix((matrix32f_t*)&cinput2);
	deleteMatrix((matrix32f_t*)&coutput1);
	deleteMatrix((matrix32f_t*)&cexpected_output);

	deleteLUT32f(&lut0);
	deleteLUT32f(&lut1);
	return ret;
}

#pragma once


typedef enum valid_functions_enum {
	/* Matrix Math (2 inputs)*/	matrixSumEnum, matrixDiffEnum, multVecByMatEnum, multMatByVecEnum,  hadamardProductEnum,
	/* Matrix Math (1 input)*/	elementwisePow2Enum, reluEnum,
	/* LUT Operations*/			sqrtLutEnum, tanhLutEnum, sigmoidLutEnum,
	/* Matrix Manipulation*/	flipEnum, extend2Enum, extend4Enum, extend8Enum,
	/* Complex In & Out */		hadamardProduct_complexEnum,
	/* Complex In, Real Out */	squaredMagnitudeEnum, angleLutEnum,
	/* Real In, Complex out*/	expiLutEnum,
	/* Complex & Real In, Complex out*/ hadamardProduct_cbrEnum,
	None
} function_t;

static const char* valid_functions_str[] = {
	/* Matrix Math (2 inputs)*/	"matrixSum", "matrixDiff", "multVecByMat", "multMatByVec", "hadamardProduct",
	/* Matrix Math (1 input)*/ 	"elementwisePow2", "relu",
	/* LUT Operations*/			"sqrtLut", "tanhLut", "sigmoidLut",
	/* Matrix Manipulation*/	"flip", "extend2", "extend4", "extend8",
	/* Complex In & Out */		"hadamardProduct_complex",
	/* Complex Inputs */		"squaredMagnitude", "angleLut",
	/* Complex Outputs*/		"expiLut",
	/* Compl. & Real In, Complex Out*/ "hadamardProduct_cbr"
};
static const uint32_t valid_function_count = 19;

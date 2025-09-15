#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "matrix.h"

#define BUFFERSIZE	(32*1024) // 32KiB

int matrixFromCSV(const char *path, size_t height, size_t width, matrix32f_t *mat);

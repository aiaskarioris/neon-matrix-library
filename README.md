# Neon Matrix Library
A C library for vector/matrix operations, accelerated using Neon intrinsics (SIMD) for Arm64 processors.
---

## Introduction
This is a C library developed for ARM64 processors with Neon SIMD instructions available. The library mainly implements linear algebra operations for vectors/matrices, with some functionality for complex numbers as well.
Additionally, the library utilizes Look-Up Tables (LUTs) to perform non-linear operations on vectors/matrices, such as square root, activation functions (e.g. tanh) or operations regarding complex numbers consisting of non-linear calculations (e.g. exponential of complex number).
Finally, functionality for implementing LSTM networks is also implemented. Training LSTM networks is outside the scope of this project.

## Motivation
This code base was developed for my thesis in the Department of Informatics and Computer Engineering, University of West Attica, titled [Study and Customization of Artificial Intelligence on SoC-FPGA for
Audio Signal Processing|] (while the first few pages are not written in english, the thesis itself is). The goal of this work was to accelerate a Neural Network DSP algorithm for audio utilizing an SoC+FPGA system. While both the software and FPGA hardware designs were implemented, due to time limitations they were not integrated. Estimates of execution time showed that the most efficient partition scheme would be to execute vector-by-matrix multiplications and most of the LSTM operation on the FPGA. However, the C code itself would theoretically achieve a speed-up of nearly x6.

## Repository Structure
This repository is split into three directories: `include`, `src` and `tests`. `include` contains all the header files of the library while `src` contains the functions' definitions. Most functions in `src` use pre-compiler directives to enable/disable the use of SIMD instructions, based on the `SERIAL` flag used during compilation. `tests` includes the source file for a number of programs used to validate the functionality of the functions in `src` as well as measure their execution times. Tests use as input data csv files located in a `csv` directory.

## Compilation
Compilation is achieved using the provided Makefile. `aarch64-linux-gnu-gcc` is required. FFTW is required. Passing `SERIAL=1` as an environment variable when calling `make` will disable SIMD instructions. Passing `BAREMETAL=1` will compile a bare-metal `.elf` program (can be stacked with `SERIAL`). By default, targets are built for LINUX with SIMD support.

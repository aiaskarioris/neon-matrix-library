GCC-FLAGS	= -march=armv8-a -mtune=cortex-a53

LIBOUT_DIR	= build/library
TEST_DIR	= tests


ifndef BAREMETAL
	CC = aarch64-linux-gnu-gcc -DLINUX
	AR = aarch64-linux-gnu-ar
else
	CC = aarch64-none-elf-gcc -DBAREMETAL
	AR = aarch64-none-elf-ar
endif

ifdef SERIAL
	OUTPUTDIR  = build/serial-tests
	GCC-FLAGS += -fno-tree-vectorize -DSERIAL
	ifdef LINUX
		FFTW-DIR   = /home/ajax/Source/Aarch64/fftw-3.3.10-aarch64-serial
	else
		ERROR = 1
	endif
else
	OUTPUTDIR  = build/tests
	GCC-FLAGS += -ftree-vectorize
	ifdef LINUX
		FFTW-DIR   = /home/ajax/Source/Aarch64/fftw-3.3.10-aarch64
	else
		FFTW-DIR   = /home/ajax/Source/Aarch64/fftw-3.3.10-aarch64-none
	endif
endif


GCC-FLAGS += -I"include/" -I$(FFTW-DIR)/include/
__FFTW-LIB   = -L$(FFTW-DIR)/lib/ -lfftw3f -lm #  -DUSE_THREADS -lfftw3f_threads -lpthread
FFTW-LIB  = -lm

ifdef DEBUG
	GCC-FLAGS += -g -DDEBUG
	LIBOUT_NAME	= libneonmatrix_debug.a
else
	LIBOUT_NAME	= libneonmatrix.a
	GCC-FLAGS += -O3
endif

SOURCE=${wildcard src/*.c}
OBJS := ${SOURCE:.c=.o}

all: config_info tests
lib: config_info ar_lib clean
tests: timing_tests functional_tests clean

functional_tests: matrix_math_test
timing_tests_n: fft_spectogram_timing_testi timing_test fc_bn_timing_test shift_scale_timing_test spectogram_timing_test lstm_timing_test output_stage_timing_test
timing_tests:  timing_test timing_test_mt fc_bn_timing_test shift_scale_timing_test spectogram_timing_test lstm_timing_test conversion_test concat_timing_test


config_info:
ifdef ERROR
	  @echo "Error: Bare metal with no SIMD support has not been compiled."
	  exit 1
endif

ifndef BAREMETAL
	  @echo "Building for Linux"
else
	  @echo "Building for Bare Metal"
endif

.c.o:
	$(CC) $(GCC-FLAGS) -c -o $@ $< $(FFTW-LIB)

ar_lib: $(OBJS)	
	$(AR) rsc $(LIBOUT_DIR)/$(LIBOUT_NAME) $(OBJS)

matrix_math_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/matrix_math_test.o $(TEST_DIR)/matrix_math_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/matrix_math_test $(OBJS) $(TEST_DIR)/matrix_math_test.o $(FFTW-LIB)

timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/timing_test.o $(TEST_DIR)/timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/timing_test $(OBJS) $(TEST_DIR)/timing_test.o $(FFTW-LIB)

timing_test_mt: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/timing_test_mt.o $(TEST_DIR)/timing_test_multithread.c $(FFTW-LIB) -lpthread -lrt -DTHREADS=4
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/timing_test_st.o $(TEST_DIR)/timing_test_multithread.c $(FFTW-LIB)  -lpthread -lrt -DTHREADS=1

	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/timing_test_mt $(OBJS) $(TEST_DIR)/timing_test_mt.o $(FFTW-LIB) -lpthread -lrt
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/timing_test_st $(OBJS) $(TEST_DIR)/timing_test_st.o $(FFTW-LIB) -lpthread -lrt


fc_bn_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/fc_bn_timing_test.o $(TEST_DIR)/fc_bn_timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/fc_bn_timing_test $(OBJS) $(TEST_DIR)/fc_bn_timing_test.o $(FFTW-LIB)

shift_scale_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/shift_scale_timing_test.o $(TEST_DIR)/shift_scale_timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/shift_scale_timing_test $(OBJS) $(TEST_DIR)/shift_scale_timing_test.o $(FFTW-LIB)

spectogram_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/spectogram_timing_test.o $(TEST_DIR)/spectogram_timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/spectogram_timing_test $(OBJS) $(TEST_DIR)/spectogram_timing_test.o $(FFTW-LIB)

fft_spectogram_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/fft_spectogram_timing_test.o $(TEST_DIR)/fft_spectogram_timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/fft_spectogram_timing_test $(OBJS) $(TEST_DIR)/fft_spectogram_timing_test.o $(FFTW-LIB)

output_stage_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/output_stage_timing_test.o $(TEST_DIR)/output_stage_timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/output_stage_timing_test $(OBJS) $(TEST_DIR)/output_stage_timing_test.o $(FFTW-LIB)

lstm_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/lstm_timing_test.o $(TEST_DIR)/lstm_timing_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/lstm_timing_test $(OBJS) $(TEST_DIR)/lstm_timing_test.o $(FFTW-LIB)

concat_timing_test: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/concat_test.o $(TEST_DIR)/concat_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/concat_test $(OBJS) $(TEST_DIR)/concat_test.o $(FFTW-LIB)


conversion_test: conv_test8bit conv_test16bit

conv_test8bit: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/conversion_test8.o $(TEST_DIR)/conversion_test.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/conversion_test8 $(OBJS) $(TEST_DIR)/conversion_test8.o $(FFTW-LIB)

conv_test16bit: $(OBJS)
	$(CC) $(GCC-FLAGS) -c -o $(TEST_DIR)/conversion_test16.o $(TEST_DIR)/conversion_test_16bit.c $(FFTW-LIB)
	$(CC) $(GCC-FLAGS)    -o $(OUTPUTDIR)/conversion_test16 $(OBJS) $(TEST_DIR)/conversion_test16.o $(FFTW-LIB)


clean:
	rm -rf src/*.o
	rm -rf tests/*.o

upload:
	scp -P22 -r ./build/tests/ 19390079@195.130.109.48:/home/19390079/neon-routines-test/

upload-serial:
	scp -P22 -r ./build/serial-tests/ 19390079@195.130.109.48:/home/19390079/neon-routines-test/


upload-src:
	scp -P22 -r ./src/*.c 19390079@195.130.109.48:/home/19390079/neon-routines-test/src

upload-lut:
	scp -P22 -C -r ./lut/ 19390079@195.130.109.48:/home/19390079/neon-routines-test/

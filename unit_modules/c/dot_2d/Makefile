CC = gcc
CUDA_PATH ?= /usr/local/cuda-12.8
CFLAGS = -O2 -Wall
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64
LIBS = -lcudart -lcublas -lopenblas -lm

out/dot_2d_gpu: dot_2d_gpu.c | out
	$(CC) $(CFLAGS) $< -o $@ $(INCLUDES) $(LDFLAGS) $(LIBS)

out:
	mkdir -p out

clean:
	rm -f out/dot_2d_gpu

.PHONY: clean

ARFLAGS= -rcv

CUDA=/usr/local/cuda
NVCC=$(CUDA)/bin/nvcc
NVCCFLAGS= -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70
CUDAFLAGS= -L$(CUDA)/lib64 -I$(CUDA)/include -lcuda -lcudart

OBJECTS= src/bsdf.o src/bvh.o src/bxdf.o src/camera.o src/intersection.o src/io.o src/math.o src/microfacet.o src/onb_math.o src/path.o src/random.o src/ray.o src/render.o src/shape.o src/texture.o src/lodepng/lodepng.o

default: libbenpt.a

%.o: %.cu %.cuh
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) -dc

device.o: $(OBJECTS)
	$(NVCC) -dlink $^ -o $@

libbenpt.a: $(OBJECTS) device.o
	ar $(ARFLAGS) $@ $^

test: test.cpp libbenpt.a
	g++ $^ -o $@ $(CUDAFLAGS) -std=c++17 -L. -lbenpt

.PHONY: clean
clean:
	rm -f src/*.o *.o test libbenpt.a *.ppm

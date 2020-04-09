ARFLAGS= -rcv

NVCCFLAGS= -arch=sm_61 -gencode=arch=compute_61,code=sm_61
CUDAFLAGS= -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcuda -lcudart

OBJECTS= ./src/bvh.o ./src/camera.o ./src/intersection.o ./src/io.o ./src/math.o ./src/primitive.o ./src/ray.o ./src/render.o

default: libbenpt.a

%.o: %.cu
	nvcc -c $^ -o $@ $(NVCCFLAGS) -dc

device.o: $(OBJECTS)
	nvcc -dlink $^ -o $@

libbenpt.a: $(OBJECTS) device.o
	ar $(ARFLAGS) $@ $^

test: test.cpp libbenpt.a
	g++ $^ -o $@ $(CUDAFLAGS) -std=c++17 -L. -lbenpt

.PHONY: clean
clean:
	rm -f src/*.o *.o test libbenpt.a

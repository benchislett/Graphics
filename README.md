# Graphics

CUDA-Accelerated Path Tracer supporting arbitrary BSDFs and multiple importance sampling.

## Output

![Cornell box with Stanford Dragon](https://raw.githubusercontent.com//benchislett/Graphics/output/output/cornell-dragon.png)

![Different Cornell box with spheres](https://raw.githubusercontent.com/benchislett/Graphics/output/output/cornell-spheres.png)

![Fireplace room render](https://raw.githubusercontent.com/benchislett/Graphics/output/output/room.png)

## Usage

Default make will build the c++ rendering library for custom use.

Building the included test file with `make test` will build and link the rendering library, and output the `test` executable
which can be configured at runtime through the commandline with the following flags:

```
-f FILE	  Specify input wavefront obj file
-h		    Print this help message.
-s N		  Specify number of samples per pixel
-x N		  Specify width in pixels
-y N		  Specify height in pixels
```

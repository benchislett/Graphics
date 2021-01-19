# Graphics

## CUDA Path Tracer - Version 2!

### The Plan

A much larger focus will be placed on performance this time around.
Performance ideas from SOTA papers will be commonplace, benchmarking will be more thorough, and performance analysis will be rigorously documented.

Some other planned changes:
- Improved pipelining [(megakernels considered harmful)](https://dl.acm.org/doi/abs/10.1145/2492045.2492060)
- More (and better) compile-time polymorphism using C++20's [concepts](https://dl.acm.org/doi/abs/10.1145/2492045.2492060)
- Specific time constraints, as in a production renderer. Specifically, 20ms / 1s / 1m benchmarks and outputs will be consistently added. Restrictions on internals, BSDF support, and bounce limits more will have to take place in order to efficiently render at these qualities.
- More pretty pictures! Lots more images in the gallery to come.

### Progress

None. Just cleaned up. Check again in a few weeks.

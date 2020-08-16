#include "trace.cuh"

#include <cfloat>

__global__ void randomInit(int xmax, curandState *state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= xmax) return;

  curand_init(1984, i, 0, &state[i]);
}

struct Intersection {
  Primitive primitive;
  float3 point;
  float3 normal;
  float time;
};

#define SIGN(a) (((a) > 0) ? 1 : (((a) < 0) ? -1 : 0))

__device__ inline bool intersect(const Ray &ray, const Primitive &primitive, Intersection *isect, float tmax = -1.f) {
  float3 edge0 = primitive.tri.b - primitive.tri.a;
  float3 edge1 = primitive.tri.c - primitive.tri.a;

  float3 h = cross(ray.direction, edge1);

  float det = dot(edge0, h);

  if (fabsf(det) < FLT_MIN) return false;

  float detInv = 1.0f / det;
  float3 s = ray.origin - primitive.tri.a;
  float u = detInv * dot(s, h);

  if (u < 0.0f || u > 1.0f) return false;

  float3 q = cross(s, edge0);
  float v = detInv * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f) return false;

  float time = detInv * dot(edge1, q);

  if (time < 0.01f) return false;
  if (tmax != -1.f && time > tmax) return false;

  isect->point = ray.origin + (ray.direction * time);
  isect->time = time;
  isect->primitive = primitive;

  float w = 1.f - u - v;
  float3 normal = primitive.tri.normalA * w + primitive.tri.normalB * u + primitive.tri.normalC * v;
  int front_face = -SIGN(dot(ray.direction, normal));
  normal = normalize(normal) * front_face;
  isect->normal = normal;

  return true;
}

__device__ inline bool intersect(const Ray &ray, const AABB &slab, float *tmin) {
  float xInv = 1.f / ray.direction.x;
  float yInv = 1.f / ray.direction.y;
  float zInv = 1.f / ray.direction.z;

  float t0 = (slab.lower.x - ray.origin.x) * xInv;
  float t1 = (slab.upper.x - ray.origin.x) * xInv;
  float t2 = (slab.lower.y - ray.origin.y) * yInv;
  float t3 = (slab.upper.y - ray.origin.y) * yInv;
  float t4 = (slab.lower.z - ray.origin.z) * zInv;
  float t5 = (slab.upper.z - ray.origin.z) * zInv;

  float tmin_ = fmaxf(fmaxf(fminf(t0, t1), fminf(t2, t3)), fminf(t4, t5));
  *tmin = tmin_;
  float tmax = fminf(fminf(fmaxf(t0, t1), fmaxf(t2, t3)), fmaxf(t4, t5));

  return (0 < tmax) && (tmin_ < tmax);
}

#include <climits>
#include <cstdio>

__device__ bool intersect(const Ray &ray, const BVH &bvh, Primitive *primitives, Intersection *isect) {
  int stack[256];
  stack[0] = INT_MAX;
  int stackIdx = 0;

  int currentNode = 0;

  bool resLeft, resRight;
  float tLeft, tRight;
  float tmax = FLT_MAX;
  Intersection tmp;

  do {
    if (currentNode < 0) {
      resLeft = intersect(ray, primitives[-currentNode - 1], &tmp);
      if (resLeft) {
        tmax = tmp.time;
        *isect = tmp;
      }
      currentNode = stack[stackIdx--];
      continue;
    }

    BVHNode current = bvh.nodes[currentNode];
    resLeft = intersect(ray, current.leftBound, &tLeft);
    resRight = intersect(ray, current.rightBound, &tRight);

    if (resLeft && resRight) {
      currentNode = current.left;
      stack[++stackIdx] = current.right;
    } else if (resLeft) {
      currentNode = current.left;
    } else if (resRight) {
      currentNode = current.right;
    } else {
      currentNode = stack[stackIdx--];
    }
  } while (currentNode != INT_MAX);

  /*
  do {

    while (currentNode >= 0 && currentNode != INT_MAX) {
      BVHNode current = bvh.nodes[currentNode];
      resLeft = intersect(ray, current.leftBound, &tLeft);
      resRight = intersect(ray, current.rightBound, &tRight);

      if (resLeft != resRight) {
        if (resLeft) currentNode = current.left;
        else currentNode = current.right;
      } else {
        if (!resLeft) {
          currentNode = stack[stackIdx--];
        } else {
          stackIdx++;
          if (tLeft < tRight) {
            currentNode = current.left;
            stack[stackIdx] = current.right;
          } else {
            currentNode = current.right;
            stack[stackIdx] = current.left;
          }
        }
      }
    }

    if (currentNode != INT_MAX) {
      resLeft = intersect(ray, primitives[-currentNode - 1], &tmp, tmax);
      if (resLeft) {
        tmax = tmp.time;
        *isect = tmp;
      }
    }

  } while (currentNode != INT_MAX);
  */

  return tmax != FLT_MAX;
}

__device__ bool intersect(const Ray &ray, const Scene &scene, Intersection *isect) {
  return intersect(ray, scene.bvh, scene.primitives, isect);
}

#define PI 3.141592653589793

__device__ float3 cosSample(float u, float v) {
  float phi = 2.f * PI * u;
  float vSqrt = sqrtf(v);
  float x = cosf(phi) * vSqrt;
  float y = sinf(phi) * vSqrt;
  float z = sqrtf(1.f - v);
  return {x, y, z};
}

__device__ float3 trace(const Ray &r, const Scene &scene, curandState *state, int depth) {
  float3 light = make_float3(0.f);
  float3 beta = make_float3(1.f);
  Ray ray = r;
  Ray ao;

  Intersection isect, isectTmp;
  bool doesHit;
  for (int bounce = 0;; bounce++) {
    doesHit = intersect(ray, scene, &isect);
    
    if (doesHit && bounce == 0 && isect.primitive.emittance > 0.f) {
      light += beta * isect.primitive.emittance;
      break;
    }

    if (!doesHit || bounce > depth) break;

    ray = {isect.point, normalize(isect.normal + cosSample(curand_uniform(state), curand_uniform(state)))};

    int lightChoice = (int)((1.f - curand_uniform(state)) * scene.lightCount);
    ao = {isect.point, scene.primitives[scene.lights[lightChoice]].sample(curand_uniform(state), curand_uniform(state)) - isect.point};
    ao.direction = normalize(ao.direction);
    doesHit = intersect(ao, scene, &isectTmp);
    float dist = dot(isectTmp.point - isect.point, isectTmp.point - isect.point);
    doesHit = doesHit && isectTmp.primitive.emittance > 0.f;
    if (doesHit && dot(isect.normal, ao.direction) > 0.f) {
      light += PI * beta * fabs(dot(isect.normal, ao.direction)) * isectTmp.primitive.emittance * isectTmp.primitive.area() / dist;
    }

    beta *= 0.7f;
  }
  return light;
}

__global__ void traceKernel(Camera camera, Scene scene, Image image, int spp, curandState *state) {
  int xid = blockDim.x * blockIdx.x + threadIdx.x;
  int yid = blockDim.y * blockIdx.y + threadIdx.y;

  if (xid >= image.width || yid >= image.height) {
    return;
  }

  int tid = yid * image.width + xid;

  curandState *localGen = &state[tid];

  float3 colour = make_float3(0.f);
  float3 colourTmp;
  float u, v;
  for (int i = 0; i < spp; i++) {
    u = ((float)xid + curand_uniform(localGen)) / (float)(image.width);
    v = ((float)yid + curand_uniform(localGen)) / (float)(image.height);

    colourTmp = trace(camera.getRay(u, v), scene, localGen, 4);
    colourTmp = fmaxf(fminf(colourTmp, make_float3(1.f)), make_float3(0.f));
    colour += colourTmp;
  }
  colour /= (float)(spp);

  image.data[tid] = colour;
}

void render(const Camera &camera, Scene &scene, Image &image) {
  int nThreads = image.width * image.height;

  cudaMalloc((void **)&image.data, nThreads * sizeof(float3));

  Primitive *primitives;
  cudaMalloc((void **)&primitives, scene.primitiveCount * sizeof(Primitive));
  cudaMemcpy(primitives, scene.primitives, scene.primitiveCount * sizeof(Primitive), cudaMemcpyHostToDevice);
  Primitive *tmpPrimitives = scene.primitives;
  scene.primitives = primitives;

  int *lights;
  cudaMalloc((void **)&lights, scene.lightCount * sizeof(int));
  cudaMemcpy(lights, scene.lights, scene.lightCount * sizeof(int), cudaMemcpyHostToDevice);
  int *tmpLights = scene.lights;
  scene.lights = lights;

  BVHNode *nodes;
  cudaMalloc((void **)&nodes, scene.bvh.nodeCount * sizeof(BVHNode));
  cudaMemcpy(nodes, scene.bvh.nodes, scene.bvh.nodeCount * sizeof(BVHNode), cudaMemcpyHostToDevice);
  BVHNode *tmpNodes = scene.bvh.nodes;
  scene.bvh.nodes = nodes;

  curandState *state;
  cudaMalloc((void **)&state, nThreads * sizeof(curandState));
  randomInit<<<nThreads / 64 + 1, 64>>>(nThreads, state);
  cudaDeviceSynchronize();

  dim3 blocks(image.width / 4 + 1, image.height / 4 + 1);
  dim3 threads(4, 4);
  traceKernel<<<blocks, threads>>>(camera, scene, image, 8, state);
  cudaDeviceSynchronize();

  cudaFree(scene.primitives);
  scene.primitives = tmpPrimitives;

  cudaFree(scene.lights);
  scene.lights = tmpLights;

  cudaFree(scene.bvh.nodes);
  scene.bvh.nodes = tmpNodes;

  float3 *data = (float3 *)malloc(image.width * image.height * sizeof(float3));
  cudaMemcpy(data, image.data, image.width * image.height * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaFree(image.data);
  image.data = data;
}


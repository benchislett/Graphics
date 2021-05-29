bool hit(float v0, float v1, float v2, float o0, float o1, float o2, float d0, float d1, float d2) {
  return v0 >= (o1 + d1);
}

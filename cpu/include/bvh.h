#ifndef __BVH__
#define __BVH__

#include "triangle.h"
#include "vec.h"

#include <stdbool.h>

typedef struct aabb_t {
    vec_t min;
    vec_t max;
} aabb_t;

typedef struct bvh_t bvh_t;
typedef struct bvh_t {
    aabb_t aabb;
    int* ts;
    int ts_len;
    bvh_t* left;
    bvh_t* right;
} bvh_t;

void bvh_traverse(bvh_t* node, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx);
void bvh_build(triangle_t* triangles, size_t triangles_len);

#endif
#ifndef __BVH__
#define __BVH__

#include "triangle.h"
#include "vec.h"

#include <stdbool.h>

typedef struct aabb_t {
    vec_t min;
    vec_t max;
} aabb_t;

typedef struct bvh_t {
    aabb_t aabb;
    int tr_len;
    // triangle_idx if leaf, else child_idx
    // (tr_len > 0 : leaf)
    union {
        int tr_idx;
        int child;
    };
} bvh_t;

void bvh_traverse(int node_idx, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx);
bool bvh_light_traverse(int node_idx, const vec_t* origin, const vec_t* dir, float* t, float light_dist2);
void bvh_build(triangle_t* triangles, size_t triangles_len);

#endif
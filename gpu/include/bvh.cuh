#ifndef __BVH__
#define __BVH__

#include "triangle.cuh"
#include "vec.cuh"

#include <stdbool.h>

struct aabb_t {
    vec_t min;
    vec_t max;
};

struct haabb_t {
    hvec_t min;
    hvec_t max;
};

struct hbvh_t {
    haabb_t aabb;
    int tr_len;
    // triangle_idx if leaf, else child_idx
    // (tr_len > 0 : leaf)
    union {
        int tr_idx;
        int child;
    };
};

struct bvh_t {
    aabb_t aabb;
    int tr_len;
    // triangle_idx if leaf, else child_idx
    // (tr_len > 0 : leaf)
    union {
        int tr_idx;
        int child;
    };
};

__device__ void bvh_traverse(int node_idx, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx);
__device__ bool bvh_light_traverse(int node_idx, const vec_t* origin, const vec_t* dir, float* t, float light_dist2);
void bvh_build(triangle_t* triangles, size_t triangles_len);

#endif
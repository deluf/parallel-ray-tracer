#ifndef __BVH__
#define __BVH__

#include "triangle.h"
#include "vec.h"

#include <stdbool.h>

typedef struct bvh_t {
    vec_t aabb_min, aabb_max;
    int left_first, tri_count;
    
} bvh_t;

bvh_t* bvh_build(triangle_t* triangles, size_t triangles_len);
void bvh_intersect(const vec_t* origin, const vec_t* dir, float* t, int* norm_dir, int* tri_idx, int node_idx);

#endif
#ifndef __BVH_CUH__
#define __BVH_CUH__

#include "triangle.cuh"
#include "vec.cuh"

#include <stdbool.h>  // bool
#include <stddef.h>   // size_t

typedef struct aabb_t 
{
    vec_t min;
    vec_t max;
}
aabb_t;

typedef struct haabb_t 
{
    hvec_t min;
    hvec_t max;
}
haabb_t;

typedef struct hbvh_t 
{
    haabb_t aabb;
    int tr_len;
    union 
    {
        int tr_idx;
        int child;
    };
}
hbvh_t;

typedef struct bvh_t 
{
    aabb_t aabb;
    int tr_len;
    union 
    {
        int tr_idx;
        int child;
    };
}
bvh_t;

/**
 * Traverses the BVH to find the closest intersection with a ray
 * 
 * @param node_idx Index of the current BVH node
 * @param origin Origin vector of the ray
 * @param dir Direction vector of the ray
 * @param norm_dir Pointer to store the normal direction index
 * @param t Pointer to store the closest intersection distance
 * @param t_idx Pointer to store the intersected triangle index
 */
__device__ void bvh_traverse(int node_idx, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx);

/**
 * Traverses the BVH to check if any triangle blocks light visibility
 * 
 * @param node_idx Index of the current BVH node
 * @param origin Origin vector of the ray
 * @param dir Direction vector of the ray
 * @param t Pointer to store the intersection distance
 * @param light_dist2 Squared distance to the light source
 * @return True if light is not blocked, false otherwise
 */
__device__ bool bvh_light_traverse(int node_idx, const vec_t* origin, const vec_t* dir, float* t, float light_dist2);

/**
 * Builds the BVH structure on the host from triangles
 * 
 * @param triangles Pointer to the list of triangles
 * @param triangles_len Total number of triangles
 */
void bvh_build(triangle_t* triangles, size_t triangles_len);

#endif

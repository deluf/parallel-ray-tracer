#ifndef __BVH__
#define __BVH__

#include <stdbool.h>  // bool

#include "triangle.h"
#include "vec.h"

/**
 * Axis-aligned bounding box
 */
typedef struct aabb_t
{
    vec_t min;
    vec_t max;
} aabb_t;

/**
 * Bounding volume hierarchy node
 */
typedef struct bvh_t
{
    aabb_t aabb;
    int tr_len;
    union
    {
        int tr_idx;
        int child;
    };
} bvh_t;

/**
 * Traverse the BVH to find the closest intersection
 * 
 * @param node_idx Index of the starting node
 * @param origin Origin of the ray
 * @param dir Direction of the ray
 * @param norm_dir Pointer to store the normal direction index
 * @param t Pointer to store the distance to the intersection
 * @param t_idx Pointer to store the intersected triangle index
 */
void bvh_traverse(int node_idx, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx);

/**
 * Traverse the BVH for shadow rays
 * 
 * @param node_idx Index of the starting node
 * @param origin Origin of the shadow ray
 * @param dir Direction of the shadow ray
 * @param t Pointer to store the distance to the intersection
 * @param light_dist2 Squared distance to the light source
 * @return True if there is a clear path to the light, false otherwise
 */
bool bvh_light_traverse(int node_idx, const vec_t* origin, const vec_t* dir, float* t, float light_dist2);

/**
 * Build a BVH from a list of triangles
 * 
 * @param triangles Array of triangles
 * @param triangles_len Number of triangles
 */
void bvh_build(triangle_t* triangles, size_t triangles_len);

#endif

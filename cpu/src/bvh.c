#include "bvh.h"
#include "raytacer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

extern triangle_t* triangles;

int* triangles_idx;
bvh_t* bvh;
int bvh_len;

extern const float EPSILON;

static void bvh_update_node_bounds(int idx);
static void bvh_subdivide(int idx);
static bool bvh_is_leaf(bvh_t* bvh){ return bvh->tri_count > 0;}
static bool bvh_intersect_AABB(const vec_t* origin, const vec_t* dir, float t, const vec_t* bmin, const vec_t* bmax);

bvh_t* bvh_build(triangle_t* triangles, size_t triangles_len){
    bvh = (bvh_t*)malloc(sizeof(bvh_t)*triangles_len*2);
    memset(bvh, 0, sizeof(bvh)*triangles_len*2);
    triangles_idx = (int*)malloc(sizeof(int)*triangles_len);
    for(int i = 0; i < triangles_len; i++){
        triangles_idx[i] = i;
        triangle_t* t = &triangles[i];
        t->centroid[0] = (t->coords[0].x + t->coords[1].x + t->coords[2].x) / 3.0f;
        t->centroid[1] = (t->coords[0].y + t->coords[1].y + t->coords[2].y) / 3.0f;
        t->centroid[2] = (t->coords[0].z + t->coords[1].z + t->coords[2].z) / 3.0f;
    }
    bvh[0].left_first = 0;
    bvh[0].tri_count = triangles_len;
    bvh_len = 1;
    bvh_update_node_bounds(0);
    bvh_subdivide(0);
    printf("BVH nodes: %d\n", bvh_len);
    return bvh;
}


static void bvh_update_node_bounds(int idx){
    bvh_t* node = &bvh[idx];
    node->aabb_min = (vec_t){1e10f, 1e10f, 1e10f};
    node->aabb_max = (vec_t){-1e10f, -1e10f, -1e10f};
    for(int first = node->left_first, i = 0; i < node->tri_count; i++){
        int leaf_idx = triangles_idx[first + i];
        triangle_t* leaf_tri = &triangles[leaf_idx];
        node->aabb_min = vec_min( &node->aabb_min, &leaf_tri->coords[0] );
        node->aabb_min = vec_min( &node->aabb_min, &leaf_tri->coords[1] );
        node->aabb_min = vec_min( &node->aabb_min, &leaf_tri->coords[2] );
        node->aabb_max = vec_max( &node->aabb_max, &leaf_tri->coords[0] );
        node->aabb_max = vec_max( &node->aabb_max, &leaf_tri->coords[1] );
        node->aabb_max = vec_max( &node->aabb_max, &leaf_tri->coords[2] );
    }
}

static void bvh_subdivide(int idx){
    // terminate recursion
    bvh_t* node = &bvh[idx];
    if (node->tri_count <= 2) return;
    // determine split axis and position
    vec_t extent = vec_sub(&node->aabb_max, &node->aabb_min);
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.arr[axis]) axis = 2;
    float splitPos = node->aabb_min.arr[axis] + extent.arr[axis] * 0.5f;
    // in-place partition
    int i = node->left_first;
    int j = i + node->tri_count - 1;
    while (i <= j)
    {
        if (triangles[triangles_idx[i]].centroid[axis] < splitPos){
            i++;
        } else {
            int tmp = triangles_idx[i];
            triangles_idx[i] = triangles_idx[j];
            triangles_idx[j] = tmp;
            j--;
        }
    }
    // abort split if one of the sides is empty
    int leftCount = i - node->left_first;
    if ((leftCount == 0) || (leftCount == node->tri_count)) return;
    // create child nodes
    int leftChildIdx = bvh_len++;
    int rightChildIdx = bvh_len++;
    bvh[leftChildIdx].left_first = node->left_first;
    bvh[leftChildIdx].tri_count = leftCount;
    bvh[rightChildIdx].left_first = i;
    bvh[rightChildIdx].tri_count = node->tri_count - leftCount;
    node->left_first = leftChildIdx;
    node->tri_count = 0;
    bvh_update_node_bounds(leftChildIdx);
    bvh_update_node_bounds(rightChildIdx);
    // recurse
    bvh_subdivide(leftChildIdx);
    bvh_subdivide(rightChildIdx);
}

void bvh_intersect(const vec_t* origin, const vec_t* dir, float* t, int* norm_dir, int* tri_idx, int node_idx){
    bvh_t* node = &bvh[node_idx];
    if(!bvh_intersect_AABB(origin, dir, *t, &node->aabb_min, &node->aabb_max)) return;
    if (bvh_is_leaf(node)){
        for(int i = 0; i < node->tri_count; i++){
            int tmp_norm;
            float tmp_t = hit_triangle(origin, dir, &triangles[triangles_idx[node->left_first + i]], &tmp_norm);
            if(tmp_t > EPSILON && tmp_t < *t){
                *t = tmp_t;
                *norm_dir = tmp_norm;
                *tri_idx = i;
            }
        }
    } else {
        bvh_intersect(origin, dir, t, norm_dir, tri_idx, node->left_first);
        bvh_intersect(origin, dir, t, norm_dir, tri_idx, node->left_first + 1);
    }
}

static bool bvh_intersect_AABB(const vec_t* origin, const vec_t* dir, float t, const vec_t* bmin, const vec_t* bmax){
    float tx1 = (bmin->x - origin->x) / dir->x, tx2 = (bmax->x - origin->x) / dir->x;
    float tmin = fminf( tx1, tx2 ), tmax = fmaxf( tx1, tx2 );
    float ty1 = (bmin->y - origin->y) / dir->y, ty2 = (bmax->y - origin->y) / dir->y;
    tmin = fmaxf( tmin, fminf( ty1, ty2 ) ), tmax = fminf( tmax, fmaxf( ty1, ty2 ) );
    float tz1 = (bmin->z - origin->z) / dir->z, tz2 = (bmax->z - origin->z) / dir->z;
    tmin = fmaxf( tmin, fminf( tz1, tz2 ) ), tmax = fminf( tmax, fmaxf( tz1, tz2 ) );
    return tmax >= tmin && tmin < t && tmax > 0;
}
#include "bvh.cuh"
#include "raytracer.cuh"
#include "light.cuh"
#include "gpu.cuh"
#include "options.cuh"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

extern triangle_t* triangles;
extern int triangles_len;

bvh_t* bvh;
int* tri_idx;
int bvh_len = 1;

extern const float EPSILON;

#define GEN_SORT(x) int sort_ ## x (const void *a, const void *b) { \
    int idx1 = *(int*)a; \
    int idx2 = *(int*)b; \
    float diff = triangles[idx1].centroid[x] - triangles[idx2].centroid[x]; \
    if(diff < 0.0f) return -1; \
    if(diff > 0.0f) return +1; \
    return 0; \
}

GEN_SORT(0)
GEN_SORT(1)
GEN_SORT(2)

typedef int (*sort_func)(const void *, const void *);
sort_func sort_algs[3] = {sort_0, sort_1, sort_2};

static vec_t aabb_center(const aabb_t* aabb){
    vec_t tmp = vec_add(&aabb->min, &aabb->max);
    return vec_mul(&tmp, 0.5f);
}

static float aabb_area(const aabb_t* aabb){
    vec_t size = vec_sub(&aabb->max, &aabb->min);
    return vec_dot(&size, &size);
}

__device__ static float aabb_intersect(const aabb_t* aabb, const vec_t* origin, const vec_t* dir){
    float tx1 = (aabb->min.x - origin->x) / dir->x;
    float tx2 = (aabb->max.x - origin->x) / dir->x;
	float tmin = fminf( tx1, tx2 );
    float tmax = fmaxf( tx1, tx2 );
	float ty1 = (aabb->min.y - origin->y) / dir->y;
    float ty2 = (aabb->max.y - origin->y) / dir->y;
	tmin = fmaxf( tmin, fminf( ty1, ty2 ) );
    tmax = fminf( tmax, fmaxf( ty1, ty2 ) );
	float tz1 = (aabb->min.z - origin->z) / dir->z;
    float tz2 = (aabb->max.z - origin->z) / dir->z;
	tmin = fmaxf( tmin, fminf( tz1, tz2 ) );
    tmax = fminf( tmax, fmaxf( tz1, tz2 ) );
    bool cond = tmax >= tmin && tmax > 0.0f;
    if(cond)
        return tmin;
	return FLT_MAX;
}

static void aabb_grow_pt(aabb_t* aabb, const vec_t* point){
    aabb->min = vec_min(&aabb->min, point);
    aabb->max = vec_max(&aabb->max, point);
}

static void aabb_grow_tr(aabb_t* aabb, int t_idx){
    triangle_t* tr = &triangles[t_idx];
    aabb_grow_pt(aabb, &tr->coords[0]);
    aabb_grow_pt(aabb, &tr->coords[1]);
    aabb_grow_pt(aabb, &tr->coords[2]);
}

static int min_stats = INT_MAX;
static int max_stats = INT_MIN;
static int sum_stats = 0;
static int count_stats = 0;

static void bvh_split(int node_idx, int depth){
    bvh_t* parent = &bvh[node_idx];
    if(bvh_len >= 2*triangles_len){
        printf("BVH SPLIT: MAX SIZE REACHED\n");
        return;
    }
    if(depth == BVH_MAX_ITER || parent->tr_len <= BVH_ELEMENT_THRESHOLD) {
        if(!parent->tr_len)
            parent->child = 0;
        #if BVH_METRICS == 1
        sum_stats += parent->tr_len;
        count_stats += 1;
        if(parent->tr_len < min_stats)
            min_stats = parent->tr_len;
        if(parent->tr_len > max_stats)
            max_stats = parent->tr_len;
        #endif
        return;
    }

    int child_idx = bvh_len;
    bvh_len += 2;

    bvh_t* left = &bvh[child_idx];
    bvh_t* right = &bvh[child_idx + 1];
    left->tr_idx = parent->tr_idx;
    left->aabb.min = vec_t{1e10f, 1e10f, 1e10f};
    left->aabb.max = vec_t{-1e10f, -1e10f, -1e10f};
    right->tr_idx = parent->tr_idx;
    right->aabb.min = vec_t{1e10f, 1e10f, 1e10f};
    right->aabb.max = vec_t{-1e10f, -1e10f, -1e10f};

    int splitAxis;
    float splitPos;
    vec_t center = aabb_center(&parent->aabb);
    vec_t size = vec_sub(&parent->aabb.max, &parent->aabb.min);

    #if BVH_HEURISTIC == 5
    aabb_t aabb_l;
    aabb_t aabb_r;
    splitAxis = 0;
    float best_score = FLT_MAX;
    for(int i = 0; i < 3; i++){
        aabb_l.max = aabb_r.max = (vec_t){FLT_MIN, FLT_MIN, FLT_MIN};
        aabb_l.min = aabb_r.min = (vec_t){FLT_MAX, FLT_MAX, FLT_MAX};
        qsort(tri_idx + parent->tr_idx, parent->tr_len, sizeof(int), sort_algs[i]);
        for(int j = parent->tr_idx; j < parent->tr_idx + parent->tr_len; j++){
            int t_idx = tri_idx[j];
            triangle_t* t = &triangles[t_idx];
            bool inA = j < parent->tr_idx + (parent->tr_len / 2);
            aabb_grow_tr(inA ? &aabb_l : &aabb_r, t_idx);
        }
        float score = (parent->tr_len/2)*aabb_area(&aabb_l) + (parent->tr_len-parent->tr_len/2)*aabb_area(&aabb_r);
        if(score < best_score){
            splitAxis = i;
            best_score = score;
        }
    }
    #endif

    #if BVH_HEURISTIC == 6
    splitAxis = 0;
    splitPos = 0;
    float best_score = FLT_MAX;
    for(int axis = 0; axis < 3; axis++){
        #if SAH_BIN_SIZE == -1
        for(int i = parent->tr_idx; i < parent->tr_idx + parent->tr_len; i++){
        #else
        for(int i = 0; i < SAH_BIN_SIZE; i++){
        #endif
            aabb_t aabb_l, aabb_r;
            aabb_l.max = aabb_r.max = vec_t{FLT_MIN, FLT_MIN, FLT_MIN};
            aabb_l.min = aabb_r.min = vec_t{FLT_MAX, FLT_MAX, FLT_MAX};
            #if SAH_BIN_SIZE == -1
            int t_idx = tri_idx[i];
            triangle_t* t = &triangles[t_idx];
            float split = t->centroid[axis];
            #else
            vec_t size = vec_sub(&parent->aabb.max, &parent->aabb.min);
            float split = parent->aabb.min.arr[axis] + size.arr[axis] * ((float)i / SAH_BIN_SIZE);
            #endif
            int cl = 0;
            int cr = 0;
            for(int j = parent->tr_idx; j < parent->tr_idx + parent->tr_len; j++){
                int t_idx = tri_idx[j];
                triangle_t* t = &triangles[t_idx];
                bool inA = t->centroid[axis] < split;
                aabb_t* aabb = inA ? &aabb_l : &aabb_r;
                aabb_grow_tr(aabb, t_idx);
                if(inA) cl++ ; else cr++;
            }
            float score = cl*aabb_area(&aabb_l) + cr*aabb_area(&aabb_r);
            if(score < best_score){
                best_score = score;
                splitAxis = axis;
                splitPos = split;
            }
        }
    }
    #endif

    #if BVH_HEURISTIC == 4
    splitAxis = 0;
    if(size.y > size.x) splitAxis = 1;
    if(size.z > size.x && size.z > size.y) splitAxis = 2;
    splitPos = center.arr[splitAxis];
    #endif

    // special behavior for median split
    #if (BVH_HEURISTIC == 4) || (BVH_HEURISTIC == 5)
    qsort(tri_idx + parent->tr_idx, parent->tr_len, sizeof(int), sort_algs[splitAxis]);

    for(int i = parent->tr_idx; i < parent->tr_idx + parent->tr_len; i++){
        int t_idx = tri_idx[i];
        triangle_t* t = &triangles[t_idx];
        bool inA = i < parent->tr_idx + (parent->tr_len / 2);
        bvh_t* child = inA ? left : right;
        aabb_grow_tr(&child->aabb, t_idx);
        child->tr_len += 1;

        if(inA){
            int swap = left->tr_idx + left->tr_len - 1;
            int tmp = tri_idx[i];
            tri_idx[i] = tri_idx[swap];
            tri_idx[swap] = tmp;
            right->tr_idx += 1;
        }
    }
    #else
    bool intersectA = false;
    bool intersectB = false;
    while(!intersectA || !intersectB){
        intersectA = false;
        intersectB = false;
        #if BVH_HEURISTIC == 6
        break;
        #elif BVH_HEURISTIC == 0
        splitAxis = 0;
        splitPos = center.arr[splitAxis];
        break;
        #elif BVH_HEURISTIC == 1
        splitAxis = 0;
        if(size.y > size.x) splitAxis = 1;
        if(size.z > size.x && size.z > size.y) splitAxis = 2;
        splitPos = center.arr[splitAxis];
        break;
        #elif BVH_HEURISTIC == 2
        splitAxis = rand() % 4;
        splitPos = center.arr[splitAxis];
        break;
        #elif BVH_HEURISTIC == 3
        splitAxis = rand() % 4;
        splitPos = center.arr[splitAxis];
        splitPos += ((float)rand()/RAND_MAX - 0.5f) * (size.arr[splitAxis]);
        

        for(int i = parent->tr_idx; i < parent->tr_idx + parent->tr_len && (!intersectA || !intersectB); i++){
            int t_idx = tri_idx[i];
            triangle_t* t = &triangles[t_idx];
            bool inA = t->centroid[splitAxis] < splitPos;
            intersectA |= inA;
            intersectB |= !inA;
        }
        #endif
    }

    for(int i = parent->tr_idx; i < parent->tr_idx + parent->tr_len; i++){
        int t_idx = tri_idx[i];
        triangle_t* t = &triangles[t_idx];
        bool inA = t->centroid[splitAxis] < splitPos;
        bvh_t* child = inA ? left : right;
        aabb_grow_tr(&child->aabb, t_idx);
        child->tr_len += 1;

        if(inA){
            int swap = left->tr_idx + left->tr_len - 1;
            int tmp = tri_idx[i];
            tri_idx[i] = tri_idx[swap];
            tri_idx[swap] = tmp;
            right->tr_idx += 1;
        }
    }
    #endif

    parent->child = child_idx;
    parent->tr_len = 0;

    bvh_split(parent->child, depth + 1);
    bvh_split(parent->child + 1, depth + 1);
}

__device__ bool bvh_light_traverse(int node_idx, const vec_t* origin, const vec_t* dir, float* t, float light_dist2){
    int stack[32];
    int stackIdx = 0;

    stack[stackIdx++] = node_idx;

    while(stackIdx){
        bvh_t node = gpu_bvh[stack[--stackIdx]];
        if(node.tr_len){
            for(int i = node.tr_idx; i < node.tr_idx + node.tr_len; i++){
                int norm_tmp;
                int idx_tmp = gpu_tri_idx[i];
                gpu_triangle_t tr = gpu_triangles[idx_tmp];
                float t_tmp = hit_triangle(origin, dir, &tr, &norm_tmp);
                if(t_tmp < *t){
                    *t = t_tmp;
                    vec_t dir_scaled = vec_mul(dir, *t);
                    vec_t intersection = vec_add(origin, &dir_scaled);
                    vec_t o_minus_i = vec_sub(origin, &intersection);
                    if(light_dist2 > vec_dot(&o_minus_i, &o_minus_i))
                        return false;
                }
            }
        } else if(node.child) {
            int near_idx = node.child;
            int far_idx = node.child + 1;
            bvh_t left = gpu_bvh[node.child];
            bvh_t right = gpu_bvh[node.child + 1];
            float near_t = aabb_intersect(&left.aabb, origin, dir);
            float far_t = aabb_intersect(&right.aabb, origin, dir);
            if(far_t < near_t){
                int tmp_idx = near_idx;
                float tmp_t = near_t;
                near_idx = far_idx;
                near_t = far_t;
                far_idx = tmp_idx;
                far_t = tmp_t;
            }
            if(far_t < *t)
                stack[stackIdx++] = far_idx;
            if(near_t < *t)
                stack[stackIdx++] = near_idx;
        }
    }

    return true;
}

__device__ void bvh_traverse(int node_idx, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx){
    int stack[32];
    int stackIdx = 0;

    stack[stackIdx++] = node_idx;

    while(stackIdx) {
        bvh_t node = gpu_bvh[stack[--stackIdx]];

        if(node.tr_len) {
            for(int i = node.tr_idx; i < node.tr_idx + node.tr_len; i++) {
                int norm_tmp;
                int idx_tmp = gpu_tri_idx[i];
                gpu_triangle_t tr = gpu_triangles[idx_tmp];
                float t_tmp = hit_triangle(origin, dir, &tr, &norm_tmp);
                if(t_tmp < *t) {
                    *t = t_tmp;
                    *norm_dir = norm_tmp;
                    *t_idx = idx_tmp;
                }
            }
        } else if(node.child) {
            int near_idx = node.child;
            int far_idx = node.child + 1;
            bvh_t left = gpu_bvh[node.child];
            bvh_t right = gpu_bvh[node.child + 1];
            float near_t = aabb_intersect(&left.aabb, origin, dir);
            float far_t = aabb_intersect(&right.aabb, origin, dir);

            if(far_t < near_t) {
                int tmp_idx = near_idx;
                float tmp_t = near_t;
                near_idx = far_idx;
                near_t = far_t;
                far_idx = tmp_idx;
                far_t = tmp_t;
            }

            if(far_t < *t)
                stack[stackIdx++] = far_idx;
            if(near_t < *t)
                stack[stackIdx++] = near_idx;
        }
    }
}

void bvh_build(triangle_t* triangles, size_t triangles_len){
    #if SEED == 0
    srand(time(NULL));
    #else
    srand(SEED);
    #endif

    if(!triangles_len){
        printf("no triangles, cannot build bvh.\n");
        exit(EXIT_FAILURE);
    }

    tri_idx = (int*)(malloc(sizeof(int)*triangles_len));
    for(int i = 0; i < triangles_len; i++)
        tri_idx[i] = i;

    bvh = (bvh_t*)malloc(sizeof(bvh_t)*2*triangles_len);
    memset(bvh, 0, sizeof(bvh_t)*2*triangles_len);
    bvh->tr_len = triangles_len;
    bvh->aabb.min = vec_t{1e10f, 1e10f, 1e10f};
    bvh->aabb.max = vec_t{-1e10f, -1e10f, -1e10f};
    for(int i = 0; i < triangles_len; i++){
        aabb_grow_tr(&bvh->aabb, i);
    }

    bvh_split(0, 0);

    #if BVH_METRICS == 1
    printf("min number of triangle: %d\n", min_stats);
    printf("max number of triangle: %d\n", max_stats);
    printf("avg number of triangle: %.2f\n", (float)sum_stats/count_stats);
    printf("number of leaf: %d\n", count_stats);
    printf("bvh size (bytes): %zd\n", sizeof(bvh_t)*bvh_len);
    #endif
}
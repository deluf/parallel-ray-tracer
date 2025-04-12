#include "bvh.h"
#include "raytracer.h"
#include "options.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

extern triangle_t* triangles;
extern size_t triangles_len;

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

static bool aabb_intersect(const aabb_t* aabb, const vec_t* origin, const vec_t* dir, float t){
    float tx1 = (aabb->min.x - origin->x) / dir->x, tx2 = (aabb->max.x - origin->x) / dir->x;
	float tmin = fminf( tx1, tx2 ), tmax = fmaxf( tx1, tx2 );
	float ty1 = (aabb->min.y - origin->y) / dir->y, ty2 = (aabb->max.y - origin->y) / dir->y;
	tmin = fmaxf( tmin, fminf( ty1, ty2 ) ), tmax = fminf( tmax, fmaxf( ty1, ty2 ) );
	float tz1 = (aabb->min.z - origin->z) / dir->z, tz2 = (aabb->max.z - origin->z) / dir->z;
	tmin = fmaxf( tmin, fminf( tz1, tz2 ) ), tmax = fminf( tmax, fmaxf( tz1, tz2 ) );
    bool cond = tmax >= tmin && tmin < t && tmax > 0;
	return cond;
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

int min_l = INT_MAX;
int max_l = INT_MIN;
int sum_l = 0;
int count_l = 0;

static void bvh_split(int node_idx, int depth){
    bvh_t* parent = &bvh[node_idx];
    if(depth == BVH_MAX_ITER || parent->tr_len <= BVH_ELEMENT_THRESHOLD) {
        if(!parent->tr_len)
            parent->child = 0;
        #if BVH_METRICS == 1
        sum_l += parent->tr_len;
        count_l += 1;
        if(parent->tr_len < min_l)
            min_l = parent->tr_len;
        if(parent->tr_len > max_l)
            max_l = parent->tr_len;
        #endif
        return;
    }

    int child_idx = bvh_len;
    bvh_len += 2;

    bvh_t* left = &bvh[child_idx];
    bvh_t* right = &bvh[child_idx + 1];
    left->tr_idx = parent->tr_idx;
    left->aabb.min = (vec_t){1e10f, 1e10f, 1e10f};
    left->aabb.max = (vec_t){-1e10f, -1e10f, -1e10f};
    right->tr_idx = parent->tr_idx;
    right->aabb.min = (vec_t){1e10f, 1e10f, 1e10f};
    right->aabb.max = (vec_t){-1e10f, -1e10f, -1e10f};

    int splitAxis;
    float splitPos;
    vec_t center = aabb_center(&parent->aabb);
    vec_t size = vec_sub(&parent->aabb.max, &parent->aabb.min);

    #if BVH_HEURISTIC == 5
    aabb_t aabb_l;
    aabb_t aabb_r;
    splitAxis = 0;
    float max_score = FLT_MIN;
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
        vec_t l_center = aabb_center(&aabb_l);
        vec_t r_center = aabb_center(&aabb_r);
        float score = vec_dist(&l_center, &r_center);
        if(score > max_score){
            splitAxis = i;
            max_score = score;
        }
    }
    #endif

    #if BVH_HEURISTIC == 4
    splitAxis = 0;
    if(size.y > size.x) splitAxis = 1;
    if(size.z > size.x && size.z > size.y) splitAxis = 2;
    splitPos = center.arr[splitAxis];
    #endif

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
        #if BVH_HEURISTIC == 0
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

bool bvh_light_traverse(int node_idx, const vec_t* origin, const vec_t* dir, float* t, float light_dist2){
    bvh_t* node = &bvh[node_idx];
    bool hit = aabb_intersect(&node->aabb, origin, dir, *t);
    if(hit){
        if(node->tr_len){
            for(int i = node->tr_idx; i < node->tr_idx + node->tr_len; i++){
                int norm_tmp;
                int idx_tmp = tri_idx[i];
                triangle_t* tr = &triangles[idx_tmp];
                float t_tmp = hit_triangle(origin, dir, tr, &norm_tmp);
                if(t_tmp < *t){
                    *t = t_tmp;
                    vec_t dir_scaled = vec_mul(dir, *t);
                    vec_t intersection = vec_add(origin, &dir_scaled);
                    vec_t o_minus_i = vec_sub(origin, &intersection);
                    if(light_dist2 > vec_dot(&o_minus_i, &o_minus_i))
                        return false;
                }
            }
        } else if(node->child) {
            bvh_light_traverse(node->child, origin, dir, t, light_dist2);
            bvh_light_traverse(node->child + 1, origin, dir, t, light_dist2);
        }
    }

    return true;
}

void bvh_traverse(int node_idx, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx){
    bvh_t* node = &bvh[node_idx];
    bool hit = aabb_intersect(&node->aabb, origin, dir, *t);
    if(hit){
        if(node->tr_len){
            for(int i = node->tr_idx; i < node->tr_idx + node->tr_len; i++){
                int norm_tmp;
                int idx_tmp = tri_idx[i];
                triangle_t* tr = &triangles[idx_tmp];
                float t_tmp = hit_triangle(origin, dir, tr, &norm_tmp);
                if(t_tmp < *t){
                    *t = t_tmp;
                    *norm_dir = norm_tmp;
                    *t_idx = idx_tmp;
                }
            }
        } else if(node->child) {
            bvh_traverse(node->child, origin, dir, norm_dir, t, t_idx);
            bvh_traverse(node->child + 1, origin, dir, norm_dir, t, t_idx);
        }
    }
}

void bvh_build(triangle_t* triangles, size_t triangles_len){
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
    bvh->aabb.min = (vec_t){1e10f, 1e10f, 1e10f};
    bvh->aabb.max = (vec_t){-1e10f, -1e10f, -1e10f};
    for(int i = 0; i < triangles_len; i++){
        aabb_grow_tr(&bvh->aabb, i);
    }

    bvh_split(0, 0);

    #if BVH_METRICS == 1
    printf("min number of triangle: %d\n", min_l);
    printf("max number of triangle: %d\n", max_l);
    printf("avg number of triangle: %.2f\n", (float)sum_l/count_l);
    printf("number of leaf: %d\n", count_l);
    printf("bvh size (bytes): %d\n", sizeof(bvh_t)*bvh_len);
    #endif
}
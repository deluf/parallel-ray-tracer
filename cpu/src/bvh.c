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

bvh_t* bvh;
int bvh_len;

extern const float EPSILON;

static vec_t aabb_center(aabb_t* aabb){
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
	return tmax >= tmin && tmin < t && tmax > 0;
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

static void bvh_add_tr(bvh_t* bvh, int t_idx){
    aabb_grow_tr(&bvh->aabb, t_idx);
    bvh->ts_len += 1;
    bvh->ts = realloc(bvh->ts, sizeof(int)*bvh->ts_len);
    bvh->ts[bvh->ts_len - 1] = t_idx;
}

int min_l = INT_MAX;
int max_l = INT_MIN;
int sum_l = 0;
int count_l = 0;

static void bvh_split(bvh_t* parent, int depth){
    if(depth == BVH_MAX_ITER || parent->ts_len <= BVH_ELEMENT_THRESHOLD) {
        #if BVH_METRICS == 1
        sum_l += parent->ts_len;
        count_l += 1;
        if(parent->ts_len < min_l)
            min_l = parent->ts_len;
        if(parent->ts_len > max_l)
            max_l = parent->ts_len;
        #endif
        return;
    }

    vec_t center = aabb_center(&parent->aabb);
    vec_t size = vec_sub(&parent->aabb.max, &parent->aabb.min);
    #if BVH_HEURISTIC == 0
    int splitAxis = 0;
    float splitPos = center.arr[splitAxis];
    #elif BVH_HEURISTIC == 1
    int splitAxis = 1;
    if(size.y > size.x) splitAxis = 1;
    if(size.z > size.x && size.z > size.y) splitAxis = 2;
    float splitPos = center.arr[splitAxis];
    #elif BVH_HEURISTIC == 2
    int splitAxis = rand() % 4;
    float splitPos = center.arr[splitAxis];
    #elif BVH_HEURISTIC == 3
    int splitAxis = rand() % 4;
    float splitPos = center.arr[splitAxis];
    splitPos += ((float)rand()/RAND_MAX - 0.5f) * (size.arr[splitAxis]);
    #endif

    parent->left = (bvh_t*)malloc(sizeof(bvh_t));
    parent->right = (bvh_t*)malloc(sizeof(bvh_t));
    memset(parent->left, 0, sizeof(bvh_t));
    memset(parent->right, 0, sizeof(bvh_t));
    parent->left->aabb.min = (vec_t){1e10f, 1e10f, 1e10f};
    parent->left->aabb.max = (vec_t){-1e10f, -1e10f, -1e10f};
    parent->right->aabb.min = (vec_t){1e10f, 1e10f, 1e10f};
    parent->right->aabb.max = (vec_t){-1e10f, -1e10f, -1e10f};

    for(int i = 0; i < parent->ts_len; i++){
        int t_idx = parent->ts[i];
        triangle_t* t = &triangles[t_idx];
        bool inA = t->centroid[splitAxis] < splitPos;
        bvh_t* child = inA ? parent->left : parent->right;
        bvh_add_tr(child, t_idx);
    }

    bvh_split(parent->left, depth + 1);
    bvh_split(parent->right, depth + 1);
}

void bvh_traverse(bvh_t* node, const vec_t* origin, const vec_t* dir, int* norm_dir, float* t, int* t_idx){
    bool hit = aabb_intersect(&node->aabb, origin, dir, *t);
    if(hit){
        if(!node->left && !node->right){
            for(int i = 0; i < node->ts_len; i++){
                int norm_tmp;
                int idx_tmp = node->ts[i];
                triangle_t* tr = &triangles[idx_tmp];
                float t_tmp = hit_triangle(origin, dir, tr, &norm_tmp);
                if(t_tmp > EPSILON && t_tmp < *t){
                    *t = t_tmp;
                    *norm_dir = norm_tmp;
                    *t_idx = idx_tmp;
                }
            }
        } else {
            bvh_traverse(node->left, origin, dir, norm_dir, t, t_idx);
            bvh_traverse(node->right, origin, dir, norm_dir, t, t_idx);
        }
    }
}

void bvh_build(triangle_t* triangles, size_t triangles_len){
    srand(time(NULL));

    bvh = (bvh_t*)malloc(sizeof(bvh_t));
    memset(bvh, 0, sizeof(bvh_t));
    bvh->aabb.min = (vec_t){1e10f, 1e10f, 1e10f};
    bvh->aabb.max = (vec_t){-1e10f, -1e10f, -1e10f};
    for(int i = 0; i < triangles_len; i++)
        bvh_add_tr(bvh, i);

    bvh_split(bvh, 0);

    #if BVH_METRICS == 1
    printf("min number of triangle: %d\n", min_l);
    printf("max number of triangle: %d\n", max_l);
    printf("avg number of triangle: %.2f\n", (float)sum_l/count_l);
    printf("number of leaf: %d\n", count_l);
    #endif
}
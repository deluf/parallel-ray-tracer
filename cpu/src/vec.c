#include "vec.h"
#include "math.h"

float vec_dot(const vec_t* v1, const vec_t* v2){
    return v1->x*v2->x + v1->y*v2->y + v2->z*v2->z;
}

float vec_dist(const vec_t* v1, const vec_t* v2){
    return sqrtf((v1->x-v2->x)*(v1->x-v2->x) + (v1->y-v2->y)*(v1->y-v2->y) + (v1->z-v2->z)*(v1->z-v2->z));
}

float vec_mag(const vec_t* v1){
    return sqrtf(v1->x*v1->x + v1->y*v1->y + v1->z*v1->z);
}

void vec_normalize(vec_t* v1){
    *v1 = vec_div(v1, vec_mag(v1));
}

vec_t vec_mul(const vec_t* v1, float val){
    return (vec_t){v1->x*val, v1->y*val, v1->z*val};
}

vec_t vec_add(const vec_t* v1, const vec_t* v2){
    return (vec_t){v1->x+v2->x, v1->y+v2->y, v1->z+v2->z};
}

vec_t vec_sub(const vec_t* v1, const vec_t* v2){
    return (vec_t){v1->x-v2->x, v1->y-v2->y, v1->z-v2->z};
}

vec_t vec_div(const vec_t* v1, float val){
    return (vec_t){v1->x/val, v1->y/val, v1->z/val};
}

vec_t vec_cross(const vec_t* v1, const vec_t* v2){
    return (vec_t){v1->y*v2->z-v1->z*v2->y, v1->z*v2->x-v1->x*v2->z, v1->x*v2->y-v1->y*v2->x};
}

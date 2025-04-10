#include "vec.h"
#include "math.h"

float vec_dot(const vec_t* v1, const vec_t* v2){
    return v1->x*v2->x + v1->y*v2->y + v1->z*v2->z;
}

float vec_dist(const vec_t* v1, const vec_t* v2){
    float dx = v1->x - v2->x;
    float dy = v1->y - v2->y;
    float dz = v1->z - v2->z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
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
    return (vec_t){
        v1->y*v2->z - v1->z*v2->y,
        v1->z*v2->x - v1->x*v2->z,
        v1->x*v2->y - v1->y*v2->x
    };
}

void vec_constrain(vec_t* v, const vec_t* min, const vec_t* max){
    v->x = fmaxf(v->x, min->x);
    v->y = fmaxf(v->y, min->y);
    v->z = fmaxf(v->z, min->z);
    v->x = fminf(v->x, max->x);
    v->y = fminf(v->y, max->y);
    v->z = fminf(v->z, max->z);
}

vec_t vec_min(const vec_t* v1, const vec_t* v2){
    return (vec_t){
        fminf(v1->x, v2->x),
        fminf(v1->y, v2->y),
        fminf(v1->z, v2->z)
    };
}
vec_t vec_max(const vec_t* v1, const vec_t* v2){
    return (vec_t){
        fmaxf(v1->x, v2->x),
        fmaxf(v1->y, v2->y),
        fmaxf(v1->z, v2->z)
    };
}
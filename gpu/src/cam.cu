#include "cam.cuh"

#include <math.h>

__host__ __device__ void cam_init(cam_t* cam, const vec_t* pos, float fov){
    cam->pos = *pos;
    cam->rot = vec_t{0, 0, 0};
    cam->fov = 1.0/tanf(fov/2.0f);
}

__host__ __device__ void cam_rotate(cam_t* cam, vec_t* p){
    cam_rotateY(cam, p);
    cam_rotateX(cam, p);
    cam_rotateZ(cam, p);
}

__host__ __device__ void cam_rotateX(cam_t* cam, vec_t* p){
    vec_t tmp = *p;
    p->y = tmp.y*cosf(cam->rot.x)-tmp.z*sinf(cam->rot.x);
    p->z = tmp.y*sinf(cam->rot.x)+tmp.z*cosf(cam->rot.x);
}

__host__ __device__ void cam_rotateY(cam_t* cam, vec_t* p){
    vec_t tmp = *p;
    p->x = tmp.x*cosf(cam->rot.y)+tmp.z*sinf(cam->rot.y);
    p->z = -tmp.x*sinf(cam->rot.y)+tmp.z*cosf(cam->rot.y);
}

__host__ __device__ void cam_rotateZ(cam_t* cam, vec_t* p){
    vec_t tmp = *p;
    p->x = tmp.x*cosf(cam->rot.z)-tmp.y*sinf(cam->rot.z);
    p->y = tmp.x*sinf(cam->rot.z)+tmp.y*cosf(cam->rot.z);
}

__host__ __device__ void cam_calculate_screen_coords(cam_t* cam, vec_t* vecs, float aspect_ratio){
    vecs[0] = vec_t{-1*aspect_ratio, cam->fov, +1};
    vecs[1] = vec_t{+1*aspect_ratio, cam->fov, +1};
    vecs[2] = vec_t{-1*aspect_ratio, cam->fov, -1};
    cam_rotate(cam, &vecs[0]);
    cam_rotate(cam, &vecs[1]);
    cam_rotate(cam, &vecs[2]);
    
    //translate using camera coordinates;
    vecs[0] = vec_add(&vecs[0], &cam->pos);
    vecs[1] = vec_add(&vecs[1], &cam->pos);
    vecs[2] = vec_add(&vecs[2], &cam->pos);
    
}

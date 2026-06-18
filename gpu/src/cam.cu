#include "cam.cuh"

#include <math.h>

__host__ __device__ void cam_init(cam_t* cam, const vec_t* pos, float fov){
    cam->pos = *pos;
    cam->rotation = vec_t{0, 0, 0};
    cam->viewport_scaling_factor = 1.0/tanf(fov/2.0f);
}

__host__ __device__ void cam_rotate(cam_t* cam, vec_t* p){
    cam_rotate_y(cam, p);
    cam_rotate_x(cam, p);
    cam_rotate_z(cam, p);
}

__host__ __device__ void cam_rotate_x(cam_t* cam, vec_t* p){
    vec_t tmp = *p;
    p->y = tmp.y*cosf(cam->rotation.x)-tmp.z*sinf(cam->rotation.x);
    p->z = tmp.y*sinf(cam->rotation.x)+tmp.z*cosf(cam->rotation.x);
}

__host__ __device__ void cam_rotate_y(cam_t* cam, vec_t* p){
    vec_t tmp = *p;
    p->x = tmp.x*cosf(cam->rotation.y)+tmp.z*sinf(cam->rotation.y);
    p->z = -tmp.x*sinf(cam->rotation.y)+tmp.z*cosf(cam->rotation.y);
}

__host__ __device__ void cam_rotate_z(cam_t* cam, vec_t* p){
    vec_t tmp = *p;
    p->x = tmp.x*cosf(cam->rotation.z)-tmp.y*sinf(cam->rotation.z);
    p->y = tmp.x*sinf(cam->rotation.z)+tmp.y*cosf(cam->rotation.z);
}

__host__ __device__ void cam_calculate_screen_coords(cam_t* cam, vec_t* vecs, float aspect_ratio){
    vecs[0] = vec_t{-1*aspect_ratio, cam->viewport_scaling_factor, +1};
    vecs[1] = vec_t{+1*aspect_ratio, cam->viewport_scaling_factor, +1};
    vecs[2] = vec_t{-1*aspect_ratio, cam->viewport_scaling_factor, -1};
    cam_rotate(cam, &vecs[0]);
    cam_rotate(cam, &vecs[1]);
    cam_rotate(cam, &vecs[2]);
    
    //translate using camera coordinates;
    vecs[0] = vec_add(&vecs[0], &cam->pos);
    vecs[1] = vec_add(&vecs[1], &cam->pos);
    vecs[2] = vec_add(&vecs[2], &cam->pos);
    
}

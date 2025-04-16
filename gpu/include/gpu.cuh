#ifndef __GPU_H__

#include "vec.cuh"
#include "triangle.cuh"
#include "light.cuh"
#include "bvh.cuh"
#include "cam.cuh"

extern __constant__ vec_t* gpu_pixels;
extern __constant__ triangle_t* gpu_triangles;
extern __constant__ int gpu_triangles_len;
extern __constant__ int* gpu_tri_idx;
extern __constant__ bvh_t* gpu_bvh;
extern __constant__ light_t* gpu_lights;
extern __constant__ int gpu_lights_len;
extern __constant__ cam_t gpu_cam;
extern __constant__ vec_t gpu_amb_light;

void render_frame();

void load_to_gpu();
void load_from_gpu();

#endif
#ifndef __GPU_H__

#include "vec.cuh"
#include "triangle.cuh"
#include "light.cuh"
#include "bvh.cuh"
#include "cam.cuh"

extern __constant__ vec_t* __restrict__ gpu_pixels;
extern __constant__ const gpu_triangle_t* __restrict__ gpu_triangles;
extern __constant__ const mat_t* __restrict__ gpu_mats;
extern __constant__ const int* __restrict__ gpu_mat_idx;
extern __constant__ const norm_t* __restrict__ gpu_norms;
extern __constant__ int gpu_triangles_len;
extern __constant__ const int* __restrict__ gpu_tri_idx;
extern __constant__ const hbvh_t* __restrict__ gpu_bvh;
extern __constant__ const light_t* __restrict__ gpu_lights;
extern __constant__ int gpu_lights_len;
extern __constant__ cam_t gpu_cam;
extern __constant__ vec_t gpu_amb_light;
extern __constant__ cudaTextureObject_t tex_tri_idx;

float render_frame(bool is_metrics, int tx, int ty);

void load_to_gpu();
void load_from_gpu();

#endif
#ifndef __GPU_CUH__
#define __GPU_CUH__

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

/**
 * Launches the CUDA rendering kernel and measures the execution time
 * 
 * @param is_metrics True to print kernel execution time, false otherwise
 * @param tx Thread block X dimension
 * @param ty Thread block Y dimension
 * @return Elapsed execution time in milliseconds
 */
float render_frame(bool is_metrics, int tx, int ty);

/**
 * Allocates device buffers and copies scene data from Host to Device
 */
void load_to_gpu(void);

/**
 * Copies pixels from Device to Host and frees allocated device buffers
 */
void load_from_gpu(void);

#endif

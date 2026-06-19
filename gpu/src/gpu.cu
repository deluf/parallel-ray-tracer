#include "gpu.cuh"
#include "options.cuh"
#include "raytracer.cuh"
#include "cam.cuh"
#include "bvh.cuh"
#include "light.cuh"
#include "triangle.cuh"
#include "vec.cuh"

#include <cuda_runtime.h>      // cudaMalloc(), cudaMemcpy(), cudaFree(), cudaEventCreate(), cudaEventRecord(), cudaEventSynchronize(), cudaEventElapsedTime(), cudaGetLastError(), cudaGetErrorString(), cudaFuncSetCacheConfig(), cudaEventDestroy(), dim3
#include <cuda_profiler_api.h>  // cudaProfilerStart(), cudaProfilerStop()
#include <stdio.h>             // printf(), fprintf(), stderr
#include <stdlib.h>            // malloc(), free(), exit(), EXIT_FAILURE

extern cam_t cam;
extern vec_t amb_light;
extern vec_t pixels[WIDTH * HEIGHT];

extern triangle_t* triangles;
extern mat_t* mats;
extern int* tri_idx;
extern size_t triangles_len;

extern bvh_t* bvh;
extern int bvh_len;

extern light_t* lights;
extern size_t lights_len; 

__constant__ vec_t* __restrict__ gpu_pixels;
__constant__ const gpu_triangle_t* __restrict__ gpu_triangles;
__constant__ const mat_t* __restrict__ gpu_mats;
__constant__ const int* __restrict__ gpu_mat_idx;
__constant__ const norm_t* __restrict__ gpu_norms;
__constant__ int gpu_triangles_len;
__constant__ const int* __restrict__ gpu_tri_idx;
__constant__ const hbvh_t* __restrict__ gpu_bvh;
__constant__ const light_t* __restrict__ gpu_lights;
__constant__ int gpu_lights_len;
__constant__ cam_t gpu_cam;
__constant__ vec_t gpu_amb_light;

/**
 * Calculates block/grid thread indices
 * 
 * @param x Output thread X coordinate
 * @param y Output thread Y coordinate
 */
__device__ static void get_idx_fast(int* x, int* y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    *x = bx * blockDim.x + tx;
    *y = by * blockDim.y + ty;
}

/**
 * Renders a single pixel by casting a ray from the camera viewport
 * 
 * @param start Start vector of the viewport block
 * @param inc_x Ray direction increment vector along the X axis
 * @param inc_y Ray direction increment vector along the Y axis
 * @param x Pixel X coordinate
 * @param y Pixel Y coordinate
 * @return Computed color vector
 */
__device__ static vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y)
{
    vec_t dir = vec_sub(start, &gpu_cam.position);
    vec_t pos_x = vec_mul(inc_x, (float)x);
    vec_t pos_y = vec_mul(inc_y, (float)y);
    dir = vec_add(&dir, &pos_x);
    dir = vec_add(&dir, &pos_y);
    return raytrace(gpu_cam.position, dir);
}

/**
 * Main GPU rendering kernel
 */
__global__ void gpu_render_frame()
{
    int x;
    int y;
    get_idx_fast(&x, &y);
    int idx = x + y * WIDTH;

    if (x >= WIDTH || y >= HEIGHT)
    {
        return;
    }

    vec_t ul = gpu_cam.viewport.top_left;
    vec_t ur = gpu_cam.viewport.top_right;
    vec_t dl = gpu_cam.viewport.bottom_left;

    vec_t inc_x = vec_sub(&ur, &ul);
    inc_x = vec_div(&inc_x, (float)WIDTH);
    vec_t inc_y = vec_sub(&dl, &ul);
    inc_y = vec_div(&inc_y, (float)HEIGHT);   
        
    vec_t out = render_pixel(&ul, &inc_x, &inc_y, x, y);
    const vec_t vec_0 = vec_t{0.0f, 0.0f, 0.0f, 0.0f};
    const vec_t vec_1 = vec_t{1.0f, 1.0f, 1.0f, 0.0f};
    vec_constrain(&out, &vec_0, &vec_1);
    gpu_pixels[idx] = out;
}

float render_frame(bool is_metrics, int tx, int ty)
{
    dim3 threads(tx, ty);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);

    cudaFuncSetCacheConfig(gpu_render_frame, cudaFuncCachePreferL1);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaProfilerStart();
    gpu_render_frame<<<blocks, threads>>>();
    cudaProfilerStop();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (is_metrics)
    {
        printf("Kernel time: %f ms\n", milliseconds);
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) 
    {
        fprintf(stderr, "Error: kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

void load_to_gpu() 
{
    gpu_triangle_t* tmp_tris = (gpu_triangle_t*)malloc(sizeof(gpu_triangle_t) * triangles_len);
    norm_t* tmp_norms = (norm_t*)malloc(sizeof(norm_t) * triangles_len);
    int* tmp_mat_idx = (int*)malloc(sizeof(int) * triangles_len);
    if (!tmp_tris || !tmp_norms || !tmp_mat_idx)
    {
        fprintf(stderr, "Error: unable to allocate temporary host buffers for gpu transfer\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < (int)triangles_len; i++)
    {
        tmp_tris[i].coords[0] = triangles[i].coords[0];
        tmp_tris[i].coords[1] = triangles[i].coords[1];
        tmp_tris[i].coords[2] = triangles[i].coords[2];
        tmp_norms[i].norm[0] = triangles[i].norm[0];
        tmp_norms[i].norm[1] = triangles[i].norm[1];
        tmp_mat_idx[i] = triangles[i].mat_idx;
    }

    vec_t* pixel_ptr = NULL;
    cudaMalloc(&pixel_ptr, sizeof(vec_t) * WIDTH * HEIGHT);
    cudaMemcpyToSymbol(gpu_pixels, &pixel_ptr, sizeof(vec_t*));

    gpu_triangle_t* triangles_ptr;
    cudaMalloc(&triangles_ptr, sizeof(gpu_triangle_t) * triangles_len);
    cudaMemcpy(triangles_ptr, tmp_tris, sizeof(gpu_triangle_t) * triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_triangles, &triangles_ptr, sizeof(gpu_triangle_t*));

    mat_t* mats_ptr;
    cudaMalloc(&mats_ptr, sizeof(mat_t) * 256);
    cudaMemcpy(mats_ptr, mats, sizeof(mat_t) * 256, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_mats, &mats_ptr, sizeof(mat_t*));

    norm_t* norm_ptr;
    cudaMalloc(&norm_ptr, sizeof(norm_t) * triangles_len);
    cudaMemcpy(norm_ptr, tmp_norms, sizeof(norm_t) * triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_norms, &norm_ptr, sizeof(norm_t*));

    int* mat_idx_ptr;
    cudaMalloc(&mat_idx_ptr, sizeof(int) * triangles_len);
    cudaMemcpy(mat_idx_ptr, tmp_mat_idx, sizeof(int) * triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_mat_idx, &mat_idx_ptr, sizeof(int*));

    light_t* lights_ptr;
    cudaMalloc(&lights_ptr, sizeof(light_t) * lights_len);
    cudaMemcpy(lights_ptr, lights, sizeof(light_t) * lights_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_lights, &lights_ptr, sizeof(light_t*));

    int* tri_ptr;
    cudaMalloc(&tri_ptr, sizeof(int) * triangles_len);
    cudaMemcpy(tri_ptr, tri_idx, sizeof(int) * triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_tri_idx, &tri_ptr, sizeof(int*));

    hbvh_t* host_hbvh = (hbvh_t*)malloc(sizeof(hbvh_t) * bvh_len);
    if (!host_hbvh)
    {
        fprintf(stderr, "Error: unable to allocate temporary bvh buffer\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < bvh_len; i++)
    {
        host_hbvh[i].tr_idx = bvh[i].tr_idx;
        host_hbvh[i].tr_len = bvh[i].tr_len;
        host_hbvh[i].aabb.min.xy = __float22half2_rn(bvh[i].aabb.min.xy);
        host_hbvh[i].aabb.min.zw = __float22half2_rn(bvh[i].aabb.min.zw);
        host_hbvh[i].aabb.max.xy = __float22half2_rn(bvh[i].aabb.max.xy);
        host_hbvh[i].aabb.max.zw = __float22half2_rn(bvh[i].aabb.max.zw);
    }

    hbvh_t* hbvh_ptr;
    cudaMalloc(&hbvh_ptr, sizeof(hbvh_t) * bvh_len);
    cudaMemcpy(hbvh_ptr, host_hbvh, sizeof(hbvh_t) * bvh_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_bvh, &hbvh_ptr, sizeof(hbvh_t*));
    free(host_hbvh);

    int trs_len = (int)triangles_len;
    int lts_len = (int)lights_len;
    cudaMemcpyToSymbol(gpu_triangles_len, &trs_len, sizeof(int));
    cudaMemcpyToSymbol(gpu_lights_len, &lts_len, sizeof(int));
    cudaMemcpyToSymbol(gpu_cam, &cam, sizeof(cam_t));
    cudaMemcpyToSymbol(gpu_amb_light, &amb_light, sizeof(vec_t));

    free(tmp_tris);
    free(tmp_mat_idx);
    free(tmp_norms);
}

void load_from_gpu() 
{
    vec_t* pixel_ptr = NULL;
    cudaMemcpyFromSymbol(&pixel_ptr, gpu_pixels, sizeof(vec_t*));
    cudaMemcpy(pixels, pixel_ptr, sizeof(vec_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaFree(pixel_ptr);

    gpu_triangle_t* trs_ptr = NULL;
    cudaMemcpyFromSymbol(&trs_ptr, gpu_triangles, sizeof(gpu_triangle_t*));
    cudaFree(trs_ptr);

    mat_t* mats_ptr = NULL;
    cudaMemcpyFromSymbol(&mats_ptr, gpu_mats, sizeof(mat_t*));
    cudaFree(mats_ptr);

    norm_t* norm_ptr = NULL;
    cudaMemcpyFromSymbol(&norm_ptr, gpu_norms, sizeof(norm_t*));
    cudaFree(norm_ptr);

    int* mat_idx_ptr = NULL;
    cudaMemcpyFromSymbol(&mat_idx_ptr, gpu_mat_idx, sizeof(int*));
    cudaFree(mat_idx_ptr);

    light_t* l_ptr = NULL;
    cudaMemcpyFromSymbol(&l_ptr, gpu_lights, sizeof(light_t*));
    cudaFree(l_ptr);

    int* tri_ptr = NULL;
    cudaMemcpyFromSymbol(&tri_ptr, gpu_tri_idx, sizeof(int*));
    cudaFree(tri_ptr);

    hbvh_t* hbvh_ptr = NULL;
    cudaMemcpyFromSymbol(&hbvh_ptr, gpu_bvh, sizeof(hbvh_t*));
    cudaFree(hbvh_ptr);
}

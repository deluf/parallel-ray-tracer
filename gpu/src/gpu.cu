#include "gpu.cuh"

#include "options.cuh"
#include "raytracer.cuh"
#include "cam.cuh"
#include "bvh.cuh"
#include "light.cuh"
#include "triangle.cuh"
#include "vec.cuh"

#include  <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>


extern cam_t cam;
extern vec_t amb_light;
extern vec_t pixels[WIDTH*HEIGHT];

extern triangle_t* triangles;
extern mat_t* mats;
extern int* tri_idx;
extern int triangles_len;

extern bvh_t* bvh;
extern int bvh_len;

extern light_t* lights;
extern int lights_len; 

__constant__ vec_t* __restrict__ gpu_pixels;
__constant__ const gpu_triangle_t* __restrict__ gpu_triangles;
__constant__ const mat_t* __restrict__ gpu_mats;
__constant__ const int* __restrict__ gpu_mat_idx;
__constant__ const norm_t* __restrict__ gpu_norms;
__constant__ int gpu_triangles_len;
__constant__ const int* __restrict__ gpu_tri_idx;
__constant__ const bvh_t* __restrict__ gpu_bvh;
__constant__ const light_t* __restrict__ gpu_lights;
__constant__ int gpu_lights_len;
__constant__ cam_t gpu_cam;
__constant__ vec_t gpu_amb_light;

__device__ void get_idx_fast(int& x, int& y){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    x = bx * blockDim.x + tx;
    y = by * blockDim.y + ty;
}


__device__ int get_idx_slow(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ vec_t render_pixel(const vec_t* start, const vec_t* inc_x, const vec_t* inc_y, int x, int y){
    vec_t dir = vec_sub(start, &gpu_cam.pos);
    vec_t pos_x = vec_mul(inc_x, x);
    vec_t pos_y = vec_mul(inc_y, y);
    dir = vec_add(&dir, &pos_x);
    dir = vec_add(&dir, &pos_y);
    return raytrace(gpu_cam.pos, dir);
}


__global__ void gpu_render_frame(){
    int idx, x, y;
    get_idx_fast(x, y);
    idx = x + y * WIDTH;
    //idx = get_idx_slow();
    //x = idx % WIDTH;
    //y = idx / WIDTH;

    if(x >= WIDTH || y >= HEIGHT)
        return;

    vec_t screen_points[3];
    cam_calculate_screen_coords(&gpu_cam, screen_points, (float)WIDTH/HEIGHT);
    vec_t ul = screen_points[0];
    vec_t ur = screen_points[1];
    vec_t dl = screen_points[2];
    vec_t inc_x = vec_sub(&ur, &ul);
    inc_x = vec_div(&inc_x, WIDTH);
    vec_t inc_y = vec_sub(&dl, &ul);
    inc_y = vec_div(&inc_y, HEIGHT);   
        
    vec_t out = render_pixel(&ul, &inc_x, &inc_y, x, y);
    const vec_t vec_0 = {0, 0, 0};
    const vec_t vec_1 = {1, 1, 1};
    vec_constrain(&out, &vec_0, &vec_1);
    gpu_pixels[idx] = out;
}

float render_frame(bool is_metrics, int tx, int ty){
    dim3 threads(tx, ty);
    dim3 blocks(WIDTH / threads.x + 1, HEIGHT / threads.y + 1);

    cudaFuncSetCacheConfig(gpu_render_frame, cudaFuncCachePreferL1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaProfilerStart();
    gpu_render_frame<<<blocks, threads>>>();
    cudaProfilerStop();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if(is_metrics)
        printf("Kernel time: %f ms\n", milliseconds);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    }

    return milliseconds;
}

void load_to_gpu() {
    gpu_triangle_t* tmp_tris = (gpu_triangle_t*)malloc(sizeof(gpu_triangle_t)*triangles_len);
    norm_t* tmp_norms = (norm_t*)malloc(sizeof(norm_t)*triangles_len);
    int* tmp_mat_idx = (int*)malloc(sizeof(int)*triangles_len);
    for(int i = 0; i < triangles_len; i++){
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
    cudaMemcpy(triangles_ptr, tmp_tris, sizeof(gpu_triangle_t)*triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_triangles, &triangles_ptr, sizeof(gpu_triangle_t*));

    mat_t* mats_ptr;
    cudaMalloc(&mats_ptr, sizeof(mat_t) * 256);
    cudaMemcpy(mats_ptr, mats, sizeof(mat_t)*256, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_mats, &mats_ptr, sizeof(mat_t*));

    norm_t* norm_ptr;
    cudaMalloc(&norm_ptr, sizeof(norm_t) * triangles_len);
    cudaMemcpy(norm_ptr, tmp_norms, sizeof(norm_t)*triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_norms, &norm_ptr, sizeof(norm_t*));

    int* mat_idx_ptr;
    cudaMalloc(&mat_idx_ptr, sizeof(int) * triangles_len);
    cudaMemcpy(mat_idx_ptr, tmp_mat_idx, sizeof(int)*triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_mat_idx, &mat_idx_ptr, sizeof(int*));

    light_t* lights_ptr;
    cudaMalloc(&lights_ptr, sizeof(light_t) * lights_len);
    cudaMemcpy(lights_ptr, lights, sizeof(light_t)*lights_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_lights, &lights_ptr, sizeof(triangle_t*));

    int* tri_ptr;
    cudaMalloc(&tri_ptr, sizeof(int)*triangles_len);
    cudaMemcpy(tri_ptr, tri_idx, sizeof(int)*triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_tri_idx, &tri_ptr, sizeof(int*));

    bvh_t* bvh_ptr;
    cudaMalloc(&bvh_ptr, sizeof(bvh_t)*bvh_len);
    cudaMemcpy(bvh_ptr, bvh, sizeof(bvh_t)*bvh_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_bvh, &bvh_ptr, sizeof(bvh_t*));

    cudaMemcpyToSymbol(gpu_triangles_len, &triangles_len, sizeof(int));
    cudaMemcpyToSymbol(gpu_lights_len, &lights_len, sizeof(int));
    cudaMemcpyToSymbol(gpu_cam, &cam, sizeof(cam_t));
    cudaMemcpyToSymbol(gpu_amb_light, &amb_light, sizeof(vec_t));

    free(tmp_tris);
    free(tmp_mat_idx);
    free(tmp_norms);
}

void load_from_gpu() {
    vec_t* pixel_ptr = NULL;
    cudaMemcpyFromSymbol(&pixel_ptr, gpu_pixels, sizeof(vec_t*));
    cudaMemcpy(pixels, pixel_ptr, sizeof(vec_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaFree(pixel_ptr);

    gpu_triangle_t* trs_ptr;
    cudaMemcpyFromSymbol(&trs_ptr, gpu_triangles, sizeof(gpu_triangle_t*));
    cudaFree(trs_ptr);

    mat_t* mats_ptr;
    cudaMemcpyFromSymbol(&mats_ptr, gpu_mats, sizeof(mat_t*));
    cudaFree(mats_ptr);

    light_t* l_ptr;
    cudaMemcpyFromSymbol(&l_ptr, gpu_lights, sizeof(light_t*));
    cudaFree(l_ptr);

    int* tri_ptr;
    cudaMemcpyFromSymbol(&tri_ptr, gpu_tri_idx, sizeof(int*));
    cudaFree(tri_ptr);

    bvh_t* bvh_ptr;
    cudaMemcpyFromSymbol(&bvh_ptr, gpu_bvh, sizeof(bvh_t*));
    cudaFree(bvh_ptr);
}
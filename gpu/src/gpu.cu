#include "gpu.cuh"

#include "options.cuh"
#include "raytracer.cuh"
#include "cam.cuh"
#include "bvh.cuh"
#include "light.cuh"
#include "triangle.cuh"
#include "vec.cuh"

#include  <cuda_runtime.h>
#include <stdio.h>


extern cam_t cam;
extern vec_t amb_light;
extern vec_t pixels[WIDTH*HEIGHT];

extern triangle_t* triangles;
extern int* tri_idx;
extern int triangles_len;

extern bvh_t* bvh;
extern int bvh_len;

extern light_t* lights;
extern int lights_len; 

__constant__ vec_t* gpu_pixels;
__constant__ triangle_t* gpu_triangles;
__constant__ int gpu_triangles_len;
__constant__ int* gpu_tri_idx;
__constant__ bvh_t* gpu_bvh;
__constant__ light_t* gpu_lights;
__constant__ int gpu_lights_len;
__constant__ cam_t gpu_cam;
__constant__ vec_t gpu_amb_light;

__device__ int get_idx(){
    /*
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int x = bx * blockDim.x + tx;
        int y = by * blockDim.y + ty;
        return x + y * WIDTH;
    */
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
    int idx = get_idx();
    int x = idx % WIDTH;
    int y = idx / WIDTH;

    if(x + y * WIDTH >= WIDTH*HEIGHT)
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

void render_frame(){
    dim3 threads(8, 8);
    dim3 blocks(WIDTH / threads.x + 1, HEIGHT / threads.y + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_render_frame<<<blocks, threads>>>();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel time: %f ms\n", milliseconds);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    }
}

void load_to_gpu() {
    vec_t* pixel_ptr = NULL;
    cudaMalloc(&pixel_ptr, sizeof(vec_t) * WIDTH * HEIGHT);
    cudaMemcpyToSymbol(gpu_pixels, &pixel_ptr, sizeof(vec_t*));

    triangle_t* triangles_ptr;
    cudaMalloc(&triangles_ptr, sizeof(triangle_t) * triangles_len);
    cudaMemcpy(triangles_ptr, triangles, sizeof(triangle_t)*triangles_len, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gpu_triangles, &triangles_ptr, sizeof(triangle_t*));

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
}

void load_from_gpu() {
    vec_t* pixel_ptr = NULL;
    cudaMemcpyFromSymbol(&pixel_ptr, gpu_pixels, sizeof(vec_t*));
    cudaMemcpy(pixels, pixel_ptr, sizeof(vec_t) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaFree(pixel_ptr);
}
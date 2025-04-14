/* scene options */

#define SCENE "car_boxed"
/*
    car_only
    car_boxed
*/

#define WIDTH (1920)
#define HEIGHT (1080)
/*
Reference (16:9):
    144p:   256 x 144
    240p:   426 x 240
    360p:   640 x 360
    480p:   854 x 480
    720p:   1280 x 720
    1080p:  1920 x 1080
    2k:     2560 x 1440
    4k:     3840 x 2160
    8k:     7680 x 4320
*/

/* number of frames to be rendered */
#define ITERATIONS 1

#define NUM_THREADS 16

/* each thread renders tiles of this size */
#define TILE_SIZE WIDTH

/* number of bounces per ray */
#define BOUNCES 4

/* bvh option */

/* max depth of bvh */
#define BVH_MAX_ITER -1

/* bvh recursion is stopped if the parent contains less than BVH_ELEMENT_THRESHOLD */
#define BVH_ELEMENT_THRESHOLD 2

/* 
    0: always axis 0
    1: axis with largest size
    2: in the middle of a random axis
    3: in a random position of a random axis
    4: median split on the largest axis
    5: median split on axis based on a distance AABB score

    option 3 should be the fastest
*/
#define BVH_HEURISTIC 3

/* BVH random split seed: only valid if either option 2 or 3 above are used */
#define SEED 1
/*
    0: seed based on time (time(NULL)) -> Useful for random tests
    1: fixed seed (BVH is always the same) -> Useful to do proper benchmarks
*/

/* print bvh metrics */
#define BVH_METRICS 1

/* raytracer */
#define USE_BVH 1

#define USE_BVH_FAST_LIGHT 1
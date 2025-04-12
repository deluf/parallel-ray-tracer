/* scene options */

#define SCENE "data"

#define WIDTH (320)
#define HEIGHT (240)

/* number of image to be rendered */
#define ITERATIONS 1

#define NUM_THREADS 16

/* number of bounces per ray */
#define BOUNCES 1

/* bvh option */

/* max depth of bvh */
#define BVH_MAX_ITER 32
/* bvh recursion is stopped if the parent contains less than BVH_ELEMENT_THRESHOLD */
#define BVH_ELEMENT_THRESHOLD 10
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

/* print bvh metrics */
#define BVH_METRICS 1

/* raytracer */
#define USE_BVH 1

#define USE_BVH_FAST_LIGHT 1
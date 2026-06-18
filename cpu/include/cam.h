#ifndef __CAM_H__
#define __CAM_H__

#include "vec.h"

/**
 * Viewport structure
 */
typedef struct viewport_t
{
    vec_t top_left;
    vec_t top_right;
    vec_t bottom_left;
} viewport_t;

/**
 * The cam_t structure represents a point camera
 * 
 *  A point in space from which the scene is observed
 */
typedef struct cam_t 
{
    vec_t position;
    viewport_t viewport;
} cam_t;

/**
 * Initializes a camera structure
 *
 * @param cam Pointer to the camera structure to initialize
 * @param position Position of the camera in the scene
 * @param rotation 3D rotation of the camera (in radians)
 * @param fov Field of view angle in radians
 * @param aspect_ratio Width to height ratio of the viewport
 * @return 0 on success, -1 on error
 */
int cam_init(cam_t* cam, const vec_t* position, const vec_t* rotation, float fov, float aspect_ratio);

#endif

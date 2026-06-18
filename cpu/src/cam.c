#include "cam.h"

#include <math.h>   // cosf(), sinf(), tanf(), M_PI
#include <stdio.h>  // fprintf(), stderr

/**
 * Rotates a point around the X axis (pitch)
 *
 * @param cam Pointer to the camera structure
 * @param p Pointer to the point to rotate
 * @param rotation Angle to rotate in radians
 */
static void cam_rotate_x(const cam_t* cam, vec_t* p, float rotation)
{
    float cos_x = cosf(rotation);
    float sin_x = sinf(rotation);
    float y = p->y * cos_x - p->z * sin_x;
    float z = p->y * sin_x + p->z * cos_x;
    p->y = y;
    p->z = z;
}

/**
 * Rotates a point around the Y axis (yaw)
 *
 * @param cam Pointer to the camera structure
 * @param p Pointer to the point to rotate
 * @param rotation Angle to rotate in radians
 */
static void cam_rotate_y(const cam_t* cam, vec_t* p, float rotation)
{
    float cos_y = cosf(rotation);
    float sin_y = sinf(rotation);
    float x = p->x * cos_y + p->z * sin_y;
    float z = -p->x * sin_y + p->z * cos_y;
    p->x = x;
    p->z = z;
}

/**
 * Rotates a point around the Z axis (roll)
 *
 * @param cam Pointer to the camera structure
 * @param p Pointer to the point to rotate
 * @param rotation Angle to rotate in radians
 */
static void cam_rotate_z(const cam_t* cam, vec_t* p, float rotation)
{
    float cos_z = cosf(rotation);
    float sin_z = sinf(rotation);
    float x = p->x * cos_z - p->y * sin_z;
    float y = p->x * sin_z + p->y * cos_z;
    p->x = x;
    p->y = y;
}

/**
 * Rotates a point around the camera's rotation axes
 * 
 *  Applies rotations in Y-X-Z order
 *
 * @param cam Pointer to the camera structure
 * @param p Pointer to the point to rotate
 * @param rotation 3D rotation of the camera (in radians)
 */
static void cam_rotate_point(const cam_t* cam, vec_t* p, const vec_t* rotation)
{
    cam_rotate_y(cam, p, rotation->x);
    cam_rotate_x(cam, p, rotation->y);
    cam_rotate_z(cam, p, rotation->z);
}

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
int cam_init(cam_t* cam, const vec_t* position, const vec_t* rotation, float fov, float aspect_ratio)
{
    if (!cam || !position || fov <= 0.0f || fov >= M_PI || aspect_ratio <= 0.0f)
    {
        fprintf(stderr, "Error: cam_init function called with bad parameters\n");
        return -1;
    }
    cam->position = *position;
    
    // How much to shrink objects with distance:
    // - the greater the camera FOV, the smaller the objects
    // - the smaller the camera FOV, the bigger the objects
    float viewport_scaling_factor = 1.0f / tanf(fov / 2.0f);

    cam->viewport.top_left = (vec_t){-aspect_ratio, viewport_scaling_factor, 1.0f};
    cam->viewport.top_right = (vec_t){aspect_ratio, viewport_scaling_factor, 1.0f};
    cam->viewport.bottom_left = (vec_t){-aspect_ratio, viewport_scaling_factor, -1.0f};

    cam_rotate_point(cam, &cam->viewport.top_left, rotation);
    cam_rotate_point(cam, &cam->viewport.top_right, rotation);
    cam_rotate_point(cam, &cam->viewport.bottom_left, rotation);
    
    cam->viewport.top_left = vec_add(&cam->viewport.top_left, &cam->position);
    cam->viewport.top_right = vec_add(&cam->viewport.top_right, &cam->position);
    cam->viewport.bottom_left = vec_add(&cam->viewport.bottom_left, &cam->position);

    return 0;
}

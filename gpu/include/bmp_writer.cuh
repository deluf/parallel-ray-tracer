#ifndef __BMP_WRITER_CUH__
#define __BMP_WRITER_CUH__

#include "vec.cuh"

/**
 * Writes a BMP image to disk from raw pixel data
 *
 * @param pixels Pointer to the pixel data (array of vec_t, RGB floats 0–1)
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param filename Path to the output BMP file
 * @return 0 on success, -1 on error
 */
int bmp_write_file(vec_t* pixels, int width, int height, const char* filename);

#endif

#include "bmp_writer.cuh"
#include "vec.cuh"

#include <stdlib.h>  // malloc(), free()
#include <string.h>  // memcpy(), memset()
#include <stdio.h>   // FILE, fopen(), fwrite(), fclose(), fprintf(), stderr
#include <stddef.h>  // size_t
#include <stdint.h>  // uint8_t, uint16_t, uint32_t

// BMP format constants
#define BMP_SIGNATURE_B 'B'
#define BMP_SIGNATURE_M 'M'
#define BMP_FILE_HEADER_SIZE 14
#define BMP_DIB_HEADER_SIZE 40
#define BMP_BITS_PER_PIXEL 32
#define BMP_PLANES 1
#define BMP_COMPRESSION_NONE 0

// BMP file header offsets
#define BMP_OFFSET_SIGNATURE 0x00
#define BMP_OFFSET_FILE_SIZE 0x02
#define BMP_OFFSET_RESERVED 0x06
#define BMP_OFFSET_DATA_START 0x0A

// BMP DIB header offsets
#define BMP_OFFSET_HEADER_SIZE 0x0E
#define BMP_OFFSET_WIDTH 0x12
#define BMP_OFFSET_HEIGHT 0x16
#define BMP_OFFSET_PLANES 0x1A
#define BMP_OFFSET_BPP 0x1C
#define BMP_OFFSET_COMPRESSION 0x1E

/**
 * Converts an RGB color stored as a vec_t
 *  into a 32-bit BGRA integer value
 *
 * @param pixel A vec_t containing r, g and b float values
 * @return The 32-bit BGRA representation of the pixel
 */
static inline uint32_t rgb_to_bgra(vec_t pixel) 
{
    uint8_t r = (uint8_t)(pixel.r * 255.0f);
    uint8_t g = (uint8_t)(pixel.g * 255.0f);
    uint8_t b = (uint8_t)(pixel.b * 255.0f);
    uint8_t a = 255;
    return b | (g << 8) | (r << 16) | (a << 24);
}

/**
 * Writes the BMP file header into the provided buffer
 *
 * @param bmp_buffer Pointer to the output BMP byte buffer
 * @param file_size Total size of the BMP file in bytes
 * @param header_size Offset where pixel data begins
 */
static void write_file_header(uint8_t* bmp_buffer, int file_size, int header_size) 
{
    bmp_buffer[BMP_OFFSET_SIGNATURE] = BMP_SIGNATURE_B;
    bmp_buffer[BMP_OFFSET_SIGNATURE + 1] = BMP_SIGNATURE_M;
    memcpy(bmp_buffer + BMP_OFFSET_FILE_SIZE, &file_size, 4);
    memset(bmp_buffer + BMP_OFFSET_RESERVED, 0, 4);
    memcpy(bmp_buffer + BMP_OFFSET_DATA_START, &header_size, 4);
}

/**
 * Writes the DIB header into the BMP buffer
 *
 * @param bmp_buffer Pointer to the output BMP byte buffer
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
static void write_dib_header(uint8_t* bmp_buffer, int width, int height) 
{
    int header_size = BMP_DIB_HEADER_SIZE;
    uint16_t planes = BMP_PLANES;
    uint16_t bpp = BMP_BITS_PER_PIXEL;
    uint32_t compression = BMP_COMPRESSION_NONE;
    
    memcpy(bmp_buffer + BMP_OFFSET_HEADER_SIZE, &header_size, 4);
    memcpy(bmp_buffer + BMP_OFFSET_WIDTH, &width, 4);
    memcpy(bmp_buffer + BMP_OFFSET_HEIGHT, &height, 4);
    memcpy(bmp_buffer + BMP_OFFSET_PLANES, &planes, 2);
    memcpy(bmp_buffer + BMP_OFFSET_BPP, &bpp, 2);
    memcpy(bmp_buffer + BMP_OFFSET_COMPRESSION, &compression, 4);
}

/**
 * Writes the pixel array to the BMP buffer in BGRA format
 *
 * @param bmp_buffer Pointer to the BMP byte buffer
 * @param header_size Offset from the start of the file where pixel data begins
 * @param pixels Pointer to the source pixel data
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param row_size Size of a single pixel row in bytes
 */
static void write_pixel_array(uint8_t* bmp_buffer, int header_size, vec_t* pixels, int width, int height, int row_size) 
{
    uint32_t* row_buffer = (uint32_t*)malloc(row_size);
    if (!row_buffer)
    {
        fprintf(stderr, "Error: unable to allocate space for the BMP row buffer\n");
        return;
    }
    
    for (int y = 0; y < height; y++) 
    {
        int src_row = height - 1 - y;
        vec_t* src_pixels = pixels + src_row * width;
        
        for (int x = 0; x < width; x++) 
        {
            row_buffer[x] = rgb_to_bgra(src_pixels[x]);
        }
        
        memcpy(bmp_buffer + header_size + y * row_size, row_buffer, row_size);
    }
    
    free(row_buffer);
}

/**
 * Creates a complete BMP image in memory from raw pixel data
 *
 * @param pixels Pointer to the pixel data
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param size Pointer to a variable that receives the total BMP size in bytes
 * @return Pointer to the allocated BMP byte buffer, or NULL on failure
 */
static uint8_t* bmp_write(vec_t* pixels, int width, int height, size_t* size)
{    
    if (BMP_BITS_PER_PIXEL != 32)
    {
        fprintf(stderr, "Error: bmp_write only works for 32 bits per pixel, got %d instead\n", BMP_BITS_PER_PIXEL);
        return NULL;
    }

    const int row_size = width * 4;
    const int pixel_array_size = row_size * height;
    const int header_size = BMP_FILE_HEADER_SIZE + BMP_DIB_HEADER_SIZE;
    const int file_size = header_size + pixel_array_size;
    
    uint8_t* bmp_buffer = (uint8_t*)malloc(file_size);
    if (!bmp_buffer) 
    {
        fprintf(stderr, "Error: unable to allocate space for the BMP buffer\n");
        return NULL;
    }
    memset(bmp_buffer, 0, file_size);
    
    write_file_header(bmp_buffer, file_size, header_size);
    write_dib_header(bmp_buffer, width, height);
    write_pixel_array(bmp_buffer, header_size, pixels, width, height, row_size);
    
    *size = file_size;
    return bmp_buffer;
}

int bmp_write_file(vec_t* pixels, int width, int height, const char* filename)
{
    if (!pixels || width <= 0 || height <= 0 || !filename) 
    {
        fprintf(stderr, "Error: bmp_write_file function called with bad parameters\n");
        return -1;
    }

    size_t img_size;
    uint8_t* img_bytes = bmp_write(pixels, width, height, &img_size);
    if (!img_bytes) 
    { 
        fprintf(stderr, "Error: unable to create the BMP buffer\n");
        return -1; 
    }
    
    FILE* img_file = fopen(filename, "wb");
    if (!img_file) 
    { 
        free(img_bytes);
        fprintf(stderr, "Error: unable to open %s\n", filename);
        return -1; 
    }
    
    if (fwrite(img_bytes, 1, img_size, img_file) != img_size) 
    {
        free(img_bytes);
        fclose(img_file);
        fprintf(stderr, "Error: unable to save BMP buffer to disk\n");
        return -1;
    }
    
    fclose(img_file);
    free(img_bytes);
    return 0;
}

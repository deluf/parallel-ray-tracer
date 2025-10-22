#include "bmp_writer.h"

#include <stdlib.h> // malloc(), free()
#include <string.h> // memcpy(), memset()
#include <stdio.h>  // FILE, fopen(), fwrite(), fclose(), perror()
#include <stddef.h> // size_t
#include <stdint.h> // uint*_t

/**
 * ================================================================================
 *                                 BMP FILE FORMAT
 * ================================================================================
 * The BMP (BitMaP) file format is the simplest, uncompressed raster image format.
 * A BMP file consists of at least three mandatory parts:
 *     1. FILE HEADER
 *     2. DIB HEADER
 *     3. PIXEL ARRAY
 * 
 * --------------------------------------------------------------------------------
 * 1. FILE HEADER (14 bytes) - Used by other software to identify the file
 * --------------------------------------------------------------------------------
 * Offset | Size | Description
 * -------+------+-----------------------------------------------------------------
 * 0x00   |  2   | Signature: 'B' (0x42), 'M' (0x4D)
 * 0x02   |  4   | File size in bytes
 * 0x06   |  4   | Reserved (set to 0 if the image is created manually)
 * 0x0A   |  4   | Address of the first byte of the pixel array
 * --------------------------------------------------------------------------------
 * 
 * --------------------------------------------------------------------------------
 * 2. DIB (Device Independent Bitmap) HEADER (BITMAPINFOHEADER, 40 bytes)
 * Contains detailed information about the image and the way pixels are encoded.
 * There are different types of DIB header as Microsoft extended it several times,
 *  the reccomended one as of 2025 is the BITMAPINFOHEADER (set the Header size to
 *  40 bytes in order to use it)
 * --------------------------------------------------------------------------------
 * Offset | Size | Description
 * -------+------+-----------------------------------------------------------------
 * 0x0E   |  4   | Header size (set to 40 bytes -> BITMAPINFOHEADER)
 * 0x12   |  4   | Image width (in pixels, signed integer)
 * 0x16   |  4   | Image height (in pixels, signed integer)
 * 0x1A   |  2   | Number of color planes (must be 1)
 * 0x1C   |  2   | Bits per pixel (commonly 24 or 32)
 * 0x1E   |  4   | Compression method (0 = none)
 * 0x22   |  4   | Image size (can be 0 if uncompressed)
 * 0x26   |  4   | Horizontal resolution (pixels per meter, signed integer)
 * 0x2A   |  4   | Vertical resolution (pixels per meter, signed integer)
 * 0x2E   |  4   | Colors in color palette (0 = default)
 * 0x32   |  4   | Important colors (0 = all)
 * --------------------------------------------------------------------------------
 * We will set the Bits per pixel to 32 (BGRA format) even tho A is always 255
 *  because pixel rows must be aligned to 4 bytes and thus it's easier to work
 *  with 4-byte pixels
 * 
 * --------------------------------------------------------------------------------
 * 3. PIXEL ARRAY
 * --------------------------------------------------------------------------------
 * BMP stores rows bottom-up: the first row in memory is the image's bottom line
 * 
 * ================================================================================
 * [Source: https://en.wikipedia.org/wiki/BMP_file_format]
 */

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

// Converts float RGB to 32-bit BGRA format
static inline uint32_t vec_to_bgra(vec_t pixel) 
{
    uint8_t r = (uint8_t)(pixel.r * 255.0f);
    uint8_t g = (uint8_t)(pixel.g * 255.0f);
    uint8_t b = (uint8_t)(pixel.b * 255.0f);
    uint8_t a = 255;
    return b | (g << 8) | (r << 16) | (a << 24);
}

static void write_file_header(uint8_t* bmp_buffer, int file_size, int header_size) 
{
    bmp_buffer[BMP_OFFSET_SIGNATURE] = BMP_SIGNATURE_B;
    bmp_buffer[BMP_OFFSET_SIGNATURE + 1] = BMP_SIGNATURE_M;
    memcpy(bmp_buffer + BMP_OFFSET_FILE_SIZE, &file_size, 4);
    memset(bmp_buffer + BMP_OFFSET_RESERVED, 0, 4);
    memcpy(bmp_buffer + BMP_OFFSET_DATA_START, &header_size, 4);
}

// Writes a DIB header of type BITMAPINFOHEADER
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

static void write_pixel_array(uint8_t* bmp_buffer, int header_size, vec_t* pixels, int width, int height, int row_size) 
{
    uint32_t* row_buffer = (uint32_t*)malloc(row_size);
    if (!row_buffer)
    {
        perror("Unable to allocate space for the BMP row buffer - No pixels will be written");
        return;
    }
    
    for (int y = 0; y < height; y++) 
    {
        // BMP stores rows bottom-up
        int src_row = height - 1 - y;
        vec_t* src_pixels = pixels + src_row * width;
        
        for (int x = 0; x < width; x++) 
        {
            row_buffer[x] = vec_to_bgra(src_pixels[x]);
        }
        
        memcpy(bmp_buffer + header_size + y * row_size, row_buffer, row_size);
    }
    
    free(row_buffer);
}

static uint8_t* bmp_write(vec_t* pixels, int width, int height, size_t* size)
{    
    if (BMP_BITS_PER_PIXEL != 32)
    {
        perror("This implementation bmp_write only works for bpp=32");
        return NULL;
    }

    const int row_size = width * 4;
    const int pixel_array_size = row_size * height;
    const int header_size = BMP_FILE_HEADER_SIZE + BMP_DIB_HEADER_SIZE;
    const int file_size = header_size + pixel_array_size;
    
    uint8_t* bmp_buffer = (uint8_t*)malloc(file_size);
    if (!bmp_buffer) 
    {
        perror("Unable to allocated space for the BMP buffer");
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
        perror("bmp_write_file function called with bad parameters");
        return -1;
    }

    size_t img_size;
    uint8_t* img_bytes = bmp_write(pixels, width, height, &img_size);
    if (!img_bytes) 
    { 
        perror("Unable to create the BMP buffer");
        return -1; 
    }
    
    FILE* img_file = fopen(filename, "wb");
    if (!img_file) 
    { 
        free(img_bytes);
        perror("Unable to open the BMP file");
        return -1; 
    }
    
    if (fwrite(img_bytes, 1, img_size, img_file) != img_size) 
    {
        free(img_bytes);
        perror("Unable to save BMP buffer to disk");
        return -1;
    }
    
    fclose(img_file);
    free(img_bytes);
    return 0;
} 

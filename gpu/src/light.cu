#include "light.cuh"

#include <stdio.h>   // FILE, fopen(), fgets(), fclose(), fprintf(), stderr, sscanf()
#include <stdlib.h>  // malloc(), realloc(), free()

light_t* load_lights(const char* filename, size_t* count)
{
    if (!filename || !count)
    {
        fprintf(stderr, "Error: load_lights function called with bad parameters\n");
        return NULL;
    }

    *count = 0;

    FILE* lights_file = fopen(filename, "r");
    if (!lights_file)
    {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return NULL;
    }

    size_t capacity = 1;
    light_t* lights = (light_t*)malloc(capacity * sizeof(light_t));
    if (!lights)
    {
        fprintf(stderr, "Error: unable to allocate memory for the lights buffer\n");
        fclose(lights_file);
        return NULL;
    }

    char line[256];
    size_t line_num = 0;

    while (fgets(line, sizeof(line), lights_file))
    {
        line_num++;

        light_t light;
        int parsed = sscanf(line, "%f %f %f %f %f %f",
                            &light.position.x, &light.position.y, &light.position.z,
                            &light.kl.r, &light.kl.g, &light.kl.b);

        if (parsed != 6)
        {
            fprintf(stderr, "Warning: skipping malformed line %zu in %s\n",
                    line_num, filename);
            continue;
        }

        if (*count >= capacity)
        {
            capacity *= 2;
            light_t* lights_expanded = (light_t*)realloc(lights, capacity * sizeof(light_t));
            if (!lights_expanded)
            {
                fprintf(stderr, "Error: unable to reallocate the lights buffer\n");
                free(lights);
                fclose(lights_file);
                return NULL;
            }
            lights = lights_expanded;
        }

        lights[*count] = light;
        (*count)++;
    }

    fclose(lights_file);

    if (*count == 0)
    {
        free(lights);
        return NULL;
    }

    if (*count < capacity)
    {
        light_t* lights_compressed = (light_t*)realloc(lights, *count * sizeof(light_t));
        if (lights_compressed)
        {
            lights = lights_compressed;
        }
    }

    return lights;
}

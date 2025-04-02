#include "sphere.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sphere_init(sphere_t* sp, float r, const vec_t* pos, const vec_t* kd, const vec_t* ks){
    sp->r = r;
    sp->pos = *pos;
    sp->kd = *kd;
    sp->ks = *ks;
}

sphere_t* sphere_load(const char* filename, size_t* size){
    sphere_t* spheres = NULL;
    *size = 0;
    FILE* fptr = fopen(filename, "r");
    if(!fptr){
        printf("cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    spheres = malloc(sizeof(sphere_t));

    char line[256];
    while (fgets(line, sizeof(line), fptr) != NULL) {
        sphere_t s;
        fscanf(fptr, "%f %f %f", &s.pos.x, &s.pos.y, &s.pos.z);
        fscanf(fptr, "%f", &s.r);
        fscanf(fptr, "%f %f %f", &s.kd.x, &s.kd.y, &s.kd.z);
        fscanf(fptr, "%f %f %f", &s.ks.x, &s.ks.y, &s.ks.z);
        *size += 1;
        spheres = realloc(spheres, sizeof(sphere_t)*(*size));
        spheres[*size-1] = s;
    }

    fclose(fptr);

    return spheres;
}
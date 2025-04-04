#include "triangle.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void triangle_init(triangle_t* t, const vec_t* a, const vec_t* b, const vec_t* c, const vec_t* ks, const vec_t* kd, const vec_t* kr){
    t->coords[0] = *a;
    t->coords[1] = *b;
    t->coords[2] = *c;
    t->ks = *ks;
    t->kd = *kd;
    t->kr = *kr;

    vec_t e1 = vec_sub(&t->coords[1], &t->coords[0]);
    vec_t e2 = vec_sub(&t->coords[2], &t->coords[0]);
    t->norm[0] = vec_cross(&e1, &e2);
    vec_normalize(&t->norm[0]);
    t->norm[1] = vec_cross(&e2, &e1);
    vec_normalize(&t->norm[1]);
}

static char** load_strings(const char* filename, int* lineCount) {
    FILE* file = fopen(filename, "r");
    if(!file){
        printf("cannot load %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    char** lines = NULL;
    char buffer[256];
    int count = 0;
    
    while(fgets(buffer, sizeof(buffer), file)) {
        lines = realloc(lines, (count + 1) * sizeof(char*));
        lines[count] = malloc(sizeof(buffer));
        strcpy(lines[count], buffer);
        count++;
    }
    
    fclose(file);
    *lineCount = count;
    return lines;
}

triangle_t* triangles_load(const char* objname, const char* mtlname, size_t* size){
    int obj_line_count, mtl_line_count;
    char** obj_lines = load_strings(objname, &obj_line_count);
    char** mtl_lines = load_strings(mtlname, &mtl_line_count);

    vec_t* vertices = NULL;
    int v_count = 0;
    vec_t ks, kd, kr;

    for (int i = 0; i < obj_line_count; i++) {
        if (obj_lines[i][0] == 'v') {
            vertices = realloc(vertices, (v_count + 1) * sizeof(vec_t));
            sscanf(obj_lines[i], "v %f %f %f", &vertices[v_count].x, &vertices[v_count].y, &vertices[v_count].z);
            v_count++;
        }
    }
    
    triangle_t* triangles = NULL;
    int count = 0;
    
    for (int i = 0; i < obj_line_count; i++) {
        if (strncmp(obj_lines[i], "usemtl", 6) == 0) {
            char material[256];
            sscanf(obj_lines[i], "usemtl %s", material);
            
            for(int j = 0; j < mtl_line_count; j++) {
                if(strncmp(mtl_lines[j], "newmtl", 6) == 0 && strstr(mtl_lines[j], material)) {
                    sscanf(mtl_lines[j + 3], "Kd %f %f %f", &kd.r, &kd.g, &kd.b);
                    sscanf(mtl_lines[j + 4], "Ks %f %f %f", &ks.r, &ks.g, &ks.b);
                    sscanf(mtl_lines[j + 5], "Kr %f %f %f", &kr.r, &kr.g, &kr.b);
                }
            }
        }
        
        if(obj_lines[i][0] == 'f') {
            int v1, v2, v3;
            sscanf(obj_lines[i], "f %d %d %d", &v1, &v2, &v3);
            triangles = realloc(triangles, (count + 1) * sizeof(triangle_t));
            triangle_init(&triangles[count], &vertices[v1 - 1], &vertices[v2 - 1], &vertices[v3 - 1], &ks, &kd, &kr);
            count++;
        }
    }
    
    free(vertices);

    for(int i = 0; i < obj_line_count; i++)
        free(obj_lines[i]);
    free(obj_lines);

    for(int i = 0; i < mtl_line_count; i++)
        free(mtl_lines[i]);
    free(mtl_lines);

    *size = count;
    return triangles;   
}

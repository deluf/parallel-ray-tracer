SRC = $(wildcard src/*.cpp src/*.cu)

all:
	nvcc -O3 -use_fast_math -Iinclude -rdc=true -dlto $(SRC) -o raytracer.exe -arch=sm_86

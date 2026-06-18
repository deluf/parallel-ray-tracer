all: default

SM_ARCH = sm_86

default:
	nvcc -O3 -use_fast_math -Iinclude -split-compile=0 -g -lineinfo -rdc=true -dlto src/*.cu -o raytracer -arch=$(SM_ARCH)

clean:
	rm -f raytracer
	rm -f *.bmp

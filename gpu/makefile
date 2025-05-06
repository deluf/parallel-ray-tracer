# Default target
all: default

# Set compiler and additional flags according to the OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    SM_ARCH = sm_75
else
    SM_ARCH = sm_86
endif
SRC = $(wildcard src/*.cpp src/*.cu)

default:
	nvcc -O3 -use_fast_math -Iinclude -rdc=true -dlto $(SRC) -o raytracer.exe -arch=$(SM_ARCH)

debug:
	nvcc -O3 -use_fast_math -Iinclude -split-compile=0 -g -lineinfo -rdc=true -dlto $(SRC) -o raytracer.exe -arch=$(SM_ARCH)

clean:
	rm -f raytracer.exe
	rm -f render.bmp

#pragma once

#include "Macro.h"


__global__ void kernel_copy_0_18(int Nx, int Ny, int Nz, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6,
                                 real *f7, real *f8, real *f9, 
                                real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                 real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, 
                                 real *f7p, real *f8p, real *f9p, 
                                 real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p);

__global__ void kernel_stream3D_0_18_forward(int Nx, int Ny, int Nz, int* bc_v, real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, 
                                            real *f10, real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                            real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, real *f7p, real *f8p, real *f9p, 
                                            real *f10p, real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p);

__global__ void kernel_wall3D_0_18(int Nx, int Ny, int Nz, int* bc_v, 
                                   real *f0, real *f1, real *f2, real *f3, real *f4, real *f5, real *f6, real *f7, real *f8, real *f9, real *f10,
                                   real *f11, real *f12, real *f13, real *f14, real *f15, real *f16, real *f17, real *f18,
                                   real *f0p, real *f1p, real *f2p, real *f3p, real *f4p, real *f5p, real *f6p, real *f7p, real *f8p, real *f9p, real *f10p, 
                                   real *f11p, real *f12p, real *f13p, real *f14p, real *f15p, real *f16p, real *f17p, real *f18p);

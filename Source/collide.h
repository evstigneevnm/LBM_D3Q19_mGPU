#pragma once

#include "Macro.h"

__global__ void kernel_collide_0_18(real delta, real *ux_old_v, real gx, real gy, real gz, int Nx, int Ny, int Nz, real omega, real *ux_v, real *uy_v, real *uz_v, real *ro_v,  int* bc_v, microscopic_variables MV_d_source, microscopic_variables MV_d_dest);
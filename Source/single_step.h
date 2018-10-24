#pragma once


#include "map.h"
#include "Macro.h"
#include "stream.h"
#include "collide.h"
#include "cuda_safe_call.h"



void run_single_step(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1, microscopic_variables MV_d2, macroscopic_variables NV_d, control_variables CV_d, real omega, real delta);
void run_single_step_streams(dim3 dimGrid, dim3 dimBlock, int Nx, int Ny, int Nz, communication_variables *COM, microscopic_variables MV_d1, microscopic_variables MV_d2, macroscopic_variables NV_d, control_variables CV_d, real omega, real delta);

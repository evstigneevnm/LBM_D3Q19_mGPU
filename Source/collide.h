/*
* This file is part of the Lattice Boltzmann multiple GPU distribution. 
(https://github.com/evstigneevnm/LBM_D3Q19_mGPU).
* Copyright (c) 2017-2018 Evstigneev Nikolay Mikhaylovitch and Ryabkov Oleg Igorevich.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 2 only.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include "Macro.h"

__global__ void kernel_collide_0_18(real delta, real *ux_old_v, real gx, real gy, real gz, int Nx, int Ny, int Nz, real omega, real *ux_v, real *uy_v, real *uz_v, real *ro_v,  int* bc_v, microscopic_variables MV_d_source, microscopic_variables MV_d_dest);
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
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

void add_perturbations_conditions(macroscopic_variables NV, int Nx, int Ny, int Nz, real amplitude);
void initial_conditions(microscopic_variables MV, macroscopic_variables NV, int Nx, int Ny, int Nz);
void get_macroscopic(microscopic_variables MV, macroscopic_variables NV, int Nx,int Ny,int Nz);
void set_boundaries(int Nx, int Ny, int Nz, control_variables CV, communication_variables COM);
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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include "Macro.h"


void write_control_file(const char f_name[], communication_variables COM, microscopic_variables MV, int Nx, int Ny, int Nz);

int read_control_file(const char f_name[], communication_variables COM, microscopic_variables MV, int Nx, int Ny, int Nz);

void write_out_file(std::string f_name, communication_variables COM, macroscopic_variables NV, control_variables CV, int Nx, int Ny, int Nz, char type, real dh, real xm=0.0, real ym=0.0, real zm=0.0);

void write_out_file_const(std::string f_name, communication_variables COM, macroscopic_variables NV, control_variables CV, int Nx, int Ny, int Nz, char type, real dh, real xm=0.0, real ym=0.0, real zm=0.0);

void write_out_pos_file(std::string f_name, int what, communication_variables COM, macroscopic_variables NV, control_variables CV, real *ux, real *uy, real *uz);

void write_out_pos_file(std::string f_name, int what, communication_variables COM, macroscopic_variables NV, control_variables CV, real *ro);
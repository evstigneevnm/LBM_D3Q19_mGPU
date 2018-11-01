#pragma once

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include "Macro.h"


void write_control_file(char f_name[], communication_variables COM, microscopic_variables MV, int Nx, int Ny, int Nz);

int read_control_file(char f_name[], communication_variables COM, microscopic_variables MV, int Nx, int Ny, int Nz);

void write_out_file(std::string f_name, communication_variables COM, macroscopic_variables NV, control_variables CV, int Nx, int Ny, int Nz, char type, real dh, real xm=0.0, real ym=0.0, real zm=0.0);

void write_out_file_const(std::string f_name, communication_variables COM, macroscopic_variables NV, control_variables CV, int Nx, int Ny, int Nz, char type, real dh, real xm=0.0, real ym=0.0, real zm=0.0);

void write_out_pos_file(std::string f_name, int what, communication_variables COM, macroscopic_variables NV, control_variables CV, real *ux, real *uy, real *uz);

void write_out_pos_file(std::string f_name, int what, communication_variables COM, macroscopic_variables NV, control_variables CV, real *ro);
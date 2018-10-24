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
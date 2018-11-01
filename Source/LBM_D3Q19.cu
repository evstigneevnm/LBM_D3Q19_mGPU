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



#include <math.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h> //for timer support
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <iostream>
#include <mpi.h>

#include "Macro.h"
#include "cuda_support.h"
#include "initial_conditions.h"
#include "IO_operations.h"
#include "single_step.h"
#include "map.h"
#include "collide.h"

void copy_microscopic_variables_host_2_device( microscopic_variables MV, microscopic_variables MV_d, int Nx, int Ny, int Nz)
{
    host_2_device_cpy<real>(MV_d.d0, MV.d0, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d1, MV.d1, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d2, MV.d2, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d3, MV.d3, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d4, MV.d4, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d5, MV.d5, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d6, MV.d6, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d7, MV.d7, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d8, MV.d8, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d9, MV.d9, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d10, MV.d10, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d11, MV.d11, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d12, MV.d12, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d13, MV.d13, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d14, MV.d14, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d15, MV.d15, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d16, MV.d16, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d17, MV.d17, Nx, Ny, Nz);
    host_2_device_cpy<real>(MV_d.d18, MV.d18, Nx, Ny, Nz);

}



void copy_microscopic_variables_device_2_host( microscopic_variables MV_d, microscopic_variables MV, int Nx, int Ny, int Nz)
{
    device_2_host_cpy<real>(MV.d0, MV_d.d0, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d1, MV_d.d1, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d2, MV_d.d2, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d3, MV_d.d3, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d4, MV_d.d4, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d5, MV_d.d5, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d6, MV_d.d6, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d7, MV_d.d7, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d8, MV_d.d8, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d9, MV_d.d9, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d10, MV_d.d10, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d11, MV_d.d11, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d12, MV_d.d12, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d13, MV_d.d13, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d14, MV_d.d14, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d15, MV_d.d15, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d16, MV_d.d16, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d17, MV_d.d17, Nx, Ny, Nz);
    device_2_host_cpy<real>(MV.d18, MV_d.d18, Nx, Ny, Nz);

}

void copy_macroscopic_variables_host_2_device( macroscopic_variables NV, macroscopic_variables NV_d, int Nx, int Ny, int Nz)
{
    host_2_device_cpy<real>(NV_d.rho, NV.rho, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.ux, NV.ux, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.uy, NV.uy, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.uz, NV.uz, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.abs_u, NV.abs_u, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.rot_x, NV.rot_x, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.rot_y, NV.rot_y, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.rot_z, NV.rot_z, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.abs_rot, NV.abs_rot, Nx, Ny, Nz);    
    host_2_device_cpy<real>(NV_d.s, NV.s, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.lk, NV.lk, Nx, Ny, Nz);
    host_2_device_cpy<real>(NV_d.H, NV.H, Nx, Ny, Nz);    

}
void copy_macroscopic_variables_device_2_host( macroscopic_variables NV_d, macroscopic_variables NV, int Nx, int Ny, int Nz)
{
    device_2_host_cpy<real>(NV.rho, NV_d.rho, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.ux, NV_d.ux, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.uy, NV_d.uy, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.uz, NV_d.uz, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.abs_u, NV_d.abs_u, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.rot_x, NV_d.rot_x, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.rot_y, NV_d.rot_y, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.rot_z, NV_d.rot_z, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.abs_rot, NV_d.abs_rot, Nx, Ny, Nz);    
    device_2_host_cpy<real>(NV.s, NV_d.s, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.lk, NV_d.lk, Nx, Ny, Nz);
    device_2_host_cpy<real>(NV.H, NV_d.H, Nx, Ny, Nz);    

}

void copy_control_variables_host_2_device( control_variables CV, control_variables CV_d, int Nx, int Ny, int Nz)
{
    host_2_device_cpy<int>(CV_d.bc, CV.bc, Nx, Ny, Nz);
    host_2_device_cpy<int>(CV_d.proc, CV.proc, Nx, Ny, Nz);
  

}

void copy_control_variables_device_2_host( control_variables CV_d, control_variables CV, int Nx, int Ny, int Nz)
{
    device_2_host_cpy<int>(CV.bc, CV_d.bc, Nx, Ny, Nz);
    device_2_host_cpy<int>(CV.proc, CV_d.proc, Nx, Ny, Nz);
  

}

int extractIntegerWords(std::string str) 
{ 
    std::stringstream ss;     

    ss << str; 
  
    std::string temp; 
    int found;
    while (!ss.eof()) { 
        
        ss >> temp; 
        if (std::stringstream(temp) >> found) 
            return found;
        temp = ""; 
    } 
    return -1;
} 

void dispatch_info_data(const char *property_tree_file_name, communication_variables *COM)
{
    if(COM->myrank==MASTER)
    {
        std::cout << "\n Using configure file: " << property_tree_file_name << std::endl;
        boost::property_tree::ptree all_properties;
        boost::property_tree::info_parser::read_info(property_tree_file_name, all_properties);
        
        int globalNx=all_properties.get<int>("Nx");
        int globalNy=all_properties.get<int>("Ny");
        int globalNz=all_properties.get<int>("Nz");
        real globalLx=all_properties.get<real>("Lx");
        real globalLy=all_properties.get<real>("Ly");
        real globalLz=all_properties.get<real>("Lz");
        real Reynolds=all_properties.get<real>("Reynolds");
        int timesteps=all_properties.get<int>("timesteps");
        char MPI_sections = all_properties.get<char>("MPI_sections");
        unsigned int MPI_fraction = all_properties.get<unsigned int>("MPI_fraction", 100); 
        for(int pr=0;pr<COM->totalrank;pr++)
        {
            std::string block_name = "Block" + std::to_string(pr);
            
            std::string L_fraction_name = block_name + ".L_fraction";
            int L_fraction = all_properties.get<int>(L_fraction_name);
            
            std::string Compute_name = block_name + ".Compute";
            std::string Compute_device = all_properties.get<std::string>(Compute_name);
            int device = 0;
            if(Compute_device == "GPU")
                device = 1;
            if(Compute_device == "PHI")
                device = 2;

            std::string Blade_ID_name = block_name + ".Blade_ID";
            int Blade_ID = all_properties.get<int>(Blade_ID_name, 0);
            
            std::string Device_PCI_ID_name = block_name + ".Device_PCI_ID";
            int Device_PCI_ID = all_properties.get<int>(Device_PCI_ID_name,-1);  //ignored for CPU
            
            std::string L_start_name = block_name + ".L_start";
            int L_start = all_properties.get<int>(L_start_name);

            int Nx=(MPI_sections=='X')?(int(globalNx*L_fraction/MPI_fraction)):globalNx;
            int Ny=(MPI_sections=='Y')?(int(globalNy*L_fraction/MPI_fraction)):globalNy;
            int Nz=(MPI_sections=='Z')?(int(globalNz*L_fraction/MPI_fraction)):globalNz;
            real dh = (MPI_sections=='X')?globalLz/globalNz:(MPI_sections=='Y')?globalLx/globalNx:(MPI_sections=='Z')?globalLy/globalNy:globalLz/globalNz;
            real Lx=(MPI_sections=='X')?(real(globalLx*L_fraction/MPI_fraction)):globalLx;
            real Ly=(MPI_sections=='Y')?(real(globalLy*L_fraction/MPI_fraction)):globalLy;
            real Lz=(MPI_sections=='Z')?(real(globalLz*L_fraction/MPI_fraction)):globalLz;
            real L0x=(MPI_sections=='X')?(real(globalLx*L_start/MPI_fraction)-2*dh*pr):0.0;
            real L0y=(MPI_sections=='Y')?(real(globalLy*L_start/MPI_fraction)-2*dh*pr):0.0;
            real L0z=(MPI_sections=='Z')?(real(globalLz*L_start/MPI_fraction)-2*dh*pr):0.0;            

            //printf("dh = %f, L0z = %f, proc = %i\n",dh, L0z, pr);

            int FaceAproc = -1;
            std::string FaceA = all_properties.get<std::string>(block_name + ".FaceA");
            if(FaceA.size()>1)
            {
                FaceAproc = extractIntegerWords(FaceA);
                FaceA="C";
            }

            int FaceBproc = -1;
            std::string FaceB = all_properties.get<std::string>(block_name + ".FaceB");
            if(FaceB.size()>1)
            {
                FaceBproc = extractIntegerWords(FaceB);
                FaceB="C";
            }

            int FaceCproc = -1;
            std::string FaceC = all_properties.get<std::string>(block_name + ".FaceC");
            if(FaceC.size()>1)
            {
                FaceCproc = extractIntegerWords(FaceC);
                FaceC="C";
            }

            int FaceDproc = -1;
            std::string FaceD = all_properties.get<std::string>(block_name + ".FaceD");
            if(FaceD.size()>1)
            {
                FaceDproc = extractIntegerWords(FaceD);
                FaceD="C";
            }

            int FaceEproc = -1;
            std::string FaceE = all_properties.get<std::string>(block_name + ".FaceE");
            if(FaceE.size()>1)
            {
                FaceEproc = extractIntegerWords(FaceE);
                FaceE="C";
            }

            int FaceFproc = -1;
            std::string FaceF = all_properties.get<std::string>(block_name + ".FaceF");
            if(FaceF.size()>1)
            {
                FaceFproc = extractIntegerWords(FaceF);
                FaceF="C";
            }


            if(pr==MASTER)
            {
                COM->Nx=Nx;
                COM->Ny=Ny;
                COM->Nz=Nz;
                COM->Lx=Lx;
                COM->Ly=Ly;
                COM->Lz=Lz;  
                COM->L0x=L0x;
                COM->L0y=L0y;
                COM->L0z=L0z;  
                COM->MPI_sections=MPI_sections;
                COM->Reynolds=Reynolds;
                COM->timesteps=timesteps;
                COM->Blade_ID = Blade_ID;
                COM->device = device;
                COM->PCI_ID = Device_PCI_ID;
                COM->device_ID = -1;
                COM->FaceA=FaceA.c_str()[0];
                COM->FaceAproc=FaceAproc;
                COM->FaceB=FaceB.c_str()[0];
                COM->FaceBproc=FaceBproc;
                COM->FaceC=FaceC.c_str()[0];
                COM->FaceCproc=FaceCproc;
                COM->FaceD=FaceD.c_str()[0];
                COM->FaceDproc=FaceDproc;                
                COM->FaceE=FaceE.c_str()[0];
                COM->FaceEproc=FaceEproc;                
                COM->FaceF=FaceF.c_str()[0];
                COM->FaceFproc=FaceFproc;                
                COM->dh=dh;
            }
            else
            {
                //send data to other processes
                MPI_Send(&Nx, 1, MPI_INT, pr, 0, MPI_COMM_WORLD); 
                MPI_Send(&Ny, 1, MPI_INT, pr, 1, MPI_COMM_WORLD); 
                MPI_Send(&Nz, 1, MPI_INT, pr, 2, MPI_COMM_WORLD); 
                MPI_Send(&Lx, 1, MPI_real, pr, 3, MPI_COMM_WORLD); 
                MPI_Send(&Ly, 1, MPI_real, pr, 4, MPI_COMM_WORLD); 
                MPI_Send(&Lz, 1, MPI_real, pr, 5, MPI_COMM_WORLD); 
                MPI_Send(&L0x, 1, MPI_real, pr, 6, MPI_COMM_WORLD); 
                MPI_Send(&L0y, 1, MPI_real, pr, 7, MPI_COMM_WORLD); 
                MPI_Send(&L0z, 1, MPI_real, pr, 8, MPI_COMM_WORLD); 
                MPI_Send(&Blade_ID, 1, MPI_INT, pr, 9, MPI_COMM_WORLD);  
                MPI_Send(&device, 1, MPI_INT, pr, 10, MPI_COMM_WORLD);
                MPI_Send(&Device_PCI_ID, 1, MPI_INT, pr, 11, MPI_COMM_WORLD);
                MPI_Send(&FaceA.c_str()[0], 1, MPI_CHAR, pr, 12, MPI_COMM_WORLD);
                MPI_Send(&FaceAproc, 1, MPI_INT, pr, 13, MPI_COMM_WORLD);
                MPI_Send(&FaceB.c_str()[0], 1, MPI_CHAR, pr, 14, MPI_COMM_WORLD);
                MPI_Send(&FaceBproc, 1, MPI_INT, pr, 15, MPI_COMM_WORLD);
                MPI_Send(&FaceC.c_str()[0], 1, MPI_CHAR, pr, 16, MPI_COMM_WORLD);
                MPI_Send(&FaceCproc, 1, MPI_INT, pr, 17, MPI_COMM_WORLD);
                MPI_Send(&FaceD.c_str()[0], 1, MPI_CHAR, pr, 18, MPI_COMM_WORLD);
                MPI_Send(&FaceDproc, 1, MPI_INT, pr, 19, MPI_COMM_WORLD);
                MPI_Send(&FaceE.c_str()[0], 1, MPI_CHAR, pr, 20, MPI_COMM_WORLD);
                MPI_Send(&FaceEproc, 1, MPI_INT, pr, 21, MPI_COMM_WORLD);
                MPI_Send(&FaceF.c_str()[0], 1, MPI_CHAR, pr, 22, MPI_COMM_WORLD);
                MPI_Send(&FaceFproc, 1, MPI_INT, pr, 23, MPI_COMM_WORLD);
                MPI_Send(&Reynolds, 1, MPI_real, pr, 24, MPI_COMM_WORLD);
                MPI_Send(&timesteps, 1, MPI_INT, pr, 25, MPI_COMM_WORLD);
                MPI_Send(&MPI_sections, 1, MPI_CHAR, pr, 26, MPI_COMM_WORLD);
                MPI_Send(&dh, 1, MPI_real, pr, 27, MPI_COMM_WORLD);

            }   


        }

    }
    else
    {
        //not master proc-s just store sent data
        MPI_Recv(&COM->Nx, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);  
        MPI_Recv(&COM->Ny, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Nz, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Lx, 1, MPI_real, MASTER, 3, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Ly, 1, MPI_real, MASTER, 4, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Lz, 1, MPI_real, MASTER, 5, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->L0x, 1, MPI_real, MASTER, 6, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->L0y, 1, MPI_real, MASTER, 7, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->L0z, 1, MPI_real, MASTER, 8, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
        MPI_Recv(&COM->Blade_ID, 1, MPI_INT, MASTER, 9, MPI_COMM_WORLD,MPI_STATUS_IGNORE);  
        MPI_Recv(&COM->device, 1, MPI_INT, MASTER, 10, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->PCI_ID, 1, MPI_INT, MASTER, 11, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceA, 1, MPI_CHAR, MASTER, 12, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceAproc, 1, MPI_INT, MASTER, 13, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceB, 1, MPI_CHAR, MASTER, 14, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceBproc, 1, MPI_INT, MASTER, 15, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceC, 1, MPI_CHAR, MASTER, 16, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceCproc, 1, MPI_INT, MASTER, 17, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceD, 1, MPI_CHAR, MASTER, 18, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceDproc, 1, MPI_INT, MASTER, 19, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceE, 1, MPI_CHAR, MASTER, 20, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceEproc, 1, MPI_INT, MASTER, 21, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->FaceF, 1, MPI_CHAR, MASTER, 22, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&COM->FaceFproc, 1, MPI_INT, MASTER, 23, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->Reynolds, 1, MPI_real, MASTER, 24, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->timesteps, 1, MPI_INT, MASTER, 25, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->MPI_sections, 1, MPI_CHAR, MASTER, 26, MPI_COMM_WORLD,MPI_STATUS_IGNORE);        
        MPI_Recv(&COM->dh, 1, MPI_real, MASTER, 27, MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char *argv[])
{
    /* code */
    communication_variables COM;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &COM.myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &COM.totalrank);
    if(argc!=2){
        if(COM.myrank==MASTER)
            std::cerr << "\nUsage: " << argv[0] << " config_file_name.info" << std::endl;
        MPI_Finalize();
        return 0;
    }

    dispatch_info_data(argv[1], &COM);
    int Nx = COM.Nx;
    int Ny = COM.Ny;
    int Nz = COM.Nz;

    int timesteps=COM.timesteps;
    real Reynolds=COM.Reynolds;
    //
    
    //CPU structures
    struct microscopic_variables MV;
    struct macroscopic_variables NV;
    struct control_variables CV;
    //GPU structures
    struct microscopic_variables MV_d1;
    struct microscopic_variables MV_d2;
    struct macroscopic_variables NV_d;
    struct control_variables CV_d;
    //select device
    if(COM.device == 1)
    {
        COM.device_ID = InitCUDA(COM.PCI_ID);
        if(COM.device_ID==-1)
        {
            printf("\n failed to select GPU device \n");
            MPI_Finalize();
            return 0;
        }
        //real * test = device_allocate<real>(1);
        cudaDeviceReset();
        std::cout << "proc " << COM.myrank << " uses device " << COM.device_ID << "\n";
    }

    //cuda streams
    cudaStreamCreate(&COM.streams[0]);

    //allocate HOST
    host_allocate_all<real>(Nx, Ny, Nz, 19, &MV.d0, &MV.d1, &MV.d2, &MV.d3, &MV.d4, &MV.d5, &MV.d6, &MV.d7,  &MV.d8,  &MV.d9,  &MV.d10, &MV.d11, &MV.d12, &MV.d13, &MV.d14, &MV.d15, &MV.d16, &MV.d17,  &MV.d18);
    host_allocate_all<real>(Nx, Ny, Nz, 12, &NV.rho, &NV.ux, &NV.uy, &NV.uz, &NV.abs_u, &NV.rot_x, &NV.rot_y, &NV.rot_z,  &NV.abs_rot,  &NV.s,  &NV.lk, &NV.H);
    host_allocate_all<int>(Nx, Ny, Nz, 2, &CV.bc, &CV.proc );

    //allocate DEVICE
    device_allocate_all<real>(Nx, Ny, Nz, 19, &MV_d1.d0, &MV_d1.d1, &MV_d1.d2, &MV_d1.d3, &MV_d1.d4, &MV_d1.d5, &MV_d1.d6, &MV_d1.d7,  &MV_d1.d8,  &MV_d1.d9,  &MV_d1.d10, &MV_d1.d11, &MV_d1.d12, &MV_d1.d13, &MV_d1.d14, &MV_d1.d15, &MV_d1.d16, &MV_d1.d17,  &MV_d1.d18);
    device_allocate_all<real>(Nx, Ny, Nz, 19, &MV_d2.d0, &MV_d2.d1, &MV_d2.d2, &MV_d2.d3, &MV_d2.d4, &MV_d2.d5, &MV_d2.d6, &MV_d2.d7,  &MV_d2.d8,  &MV_d2.d9,  &MV_d2.d10, &MV_d2.d11, &MV_d2.d12, &MV_d2.d13, &MV_d2.d14, &MV_d2.d15, &MV_d2.d16, &MV_d2.d17,  &MV_d2.d18);    
    device_allocate_all<real>(Nx, Ny, Nz, 12, &NV_d.rho, &NV_d.ux, &NV_d.uy, &NV_d.uz, &NV_d.abs_u, &NV_d.rot_x, &NV_d.rot_y, &NV_d.rot_z,  &NV_d.abs_rot,  &NV_d.s,  &NV_d.lk, &NV_d.H);
    device_allocate_all<int>(Nx, Ny, Nz, 2, &CV_d.bc, &CV_d.proc );

    //Allocate communication blocks
    allocate_blocks(Nx, Ny, Nz, &COM);

    set_boundaries(Nx, Ny, Nz, CV, COM);
    if(read_control_file("general_out_1.dat", COM, MV, Nx, Ny, Nz)==0)
        initial_conditions(MV, NV, Nx, Ny, Nz);
    else
        get_macroscopic(MV, NV, Nx, Ny, Nz);


    //copy from HOST to DEVICE
    copy_microscopic_variables_host_2_device( MV, MV_d1, Nx, Ny, Nz);
    copy_microscopic_variables_host_2_device( MV, MV_d2, Nx, Ny, Nz);
    copy_macroscopic_variables_host_2_device( NV, NV_d, Nx, Ny, Nz);
    copy_control_variables_host_2_device(CV, CV_d, Nx, Ny, Nz);
    
    //set device grid
    unsigned int k1, k2 ;
        // step 1: compute # of threads per block
    unsigned int nthreads = BLOCK_DIM_X * BLOCK_DIM_Y ;
        // step 2: compute # of blocks needed
    unsigned int nblocks = ( Nx*Ny*Nz + nthreads -1 )/nthreads ;
        // step 3: find grid's configuration
    real db_nblocks = (real)nblocks ;
    k1 = (unsigned int) floor( sqrt(db_nblocks) ) ;
    k2 = (unsigned int) ceil( db_nblocks/((real)k1)) ;

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 dimGrid( k2, k1, 1 );

    real delta=0.1;
    
    real tau=real(6.0/3.0/Reynolds+0.5);
    
    printf("tau=%lf\n",(double)tau);
    real omega=1.0/tau;


    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(int t=0;t<=timesteps;t++)
    {
        //run_single_step(dimGrid, dimBlock, Nx, Ny, Nz, &COM, MV_d1, MV_d2,  NV_d,  CV_d, omega, delta);
        run_single_step_streams(dimGrid, dimBlock, Nx, Ny, Nz, &COM, MV_d1, MV_d2,  NV_d,  CV_d, omega, delta);

        if((t%1000)==0){
            printf(" [%.03lf%%]    \r",(double)(real(t)*100.0/real(timesteps)));
            fflush(stdout);
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    gettimeofday(&end, NULL);
    real etime=((end.tv_sec-start.tv_sec)*1000000u+(end.tv_usec-start.tv_usec))/1.0E6;
    printf("\n\nWall time:%lfsec\n",(double)etime); 


    //copy from DEVICE to HOST
    copy_microscopic_variables_device_2_host( MV_d1, MV, Nx, Ny, Nz);
    copy_macroscopic_variables_device_2_host( NV_d, NV,  Nx,  Ny,  Nz);
    copy_control_variables_device_2_host( CV_d, CV, Nx, Ny, Nz);
    
    //cuda streams
    cudaStreamDestroy(COM.streams[0]);

    //free DEVICE memroy
    device_deallocate_all<real>(19, MV_d1.d0, MV_d1.d1, MV_d1.d2, MV_d1.d3, MV_d1.d4, MV_d1.d5, MV_d1.d6, MV_d1.d7,  MV_d1.d8,  MV_d1.d9,  MV_d1.d10, MV_d1.d11, MV_d1.d12, MV_d1.d13, MV_d1.d14, MV_d1.d15, MV_d1.d16, MV_d1.d17,  MV_d1.d18);
    device_deallocate_all<real>(19, MV_d2.d0, MV_d2.d1, MV_d2.d2, MV_d2.d3, MV_d2.d4, MV_d2.d5, MV_d2.d6, MV_d2.d7,  MV_d2.d8,  MV_d2.d9,  MV_d2.d10, MV_d2.d11, MV_d2.d12, MV_d2.d13, MV_d2.d14, MV_d2.d15, MV_d2.d16, MV_d2.d17,  MV_d2.d18);    
    device_deallocate_all<real>(12, NV_d.rho, NV_d.ux, NV_d.uy, NV_d.uz, NV_d.abs_u, NV_d.rot_x, NV_d.rot_y, NV_d.rot_z,  NV_d.abs_rot,  NV_d.s,  NV_d.lk, NV_d.H);
    device_deallocate_all<int>(2, CV_d.bc, CV_d.proc );
    //deallocate communication blocks
    deallocate_blocks(Nx, Ny, Nz, &COM);

    //write results to disk
    //write_out_file("vectors_out", COM, NV, CV, Nx, Ny, Nz, 'v', COM.dh, COM.L0x, COM.L0y, COM.L0z);
    //write_out_file("density_out", COM, NV, CV, Nx, Ny, Nz, 'p', COM.dh, COM.L0x, COM.L0y, COM.L0z);
    //write_out_file_const("boundary_out", COM, NV, CV, Nx, Ny, Nz, 'b', COM.dh, COM.L0x, COM.L0y, COM.L0z);
    //write_out_file_const("density_out", COM, NV, CV, Nx, Ny, Nz, 'p', COM.dh, COM.L0x, COM.L0y, COM.L0z);

    write_out_pos_file("vectors_out.pos", 2, COM, NV, CV, NV.ux, NV.uy, NV.uz);
    write_control_file("general_out_1.dat", COM, MV, Nx, Ny, Nz);

    //Free CPU meemory
    host_deallocate_all<real>(19, MV.d0, MV.d1, MV.d2, MV.d3, MV.d4, MV.d5, MV.d6, MV.d7,  MV.d8,  MV.d9,  MV.d10, MV.d11, MV.d12, MV.d13, MV.d14, MV.d15, MV.d16, MV.d17, MV.d18);
    host_deallocate_all<real>(12, NV.rho, NV.ux, NV.uy, NV.uz, NV.abs_u, NV.rot_x, NV.rot_y, NV.rot_z,  NV.abs_rot,  NV.s,  NV.lk, NV.H);
    host_deallocate_all<int>(2, CV.bc, CV.proc);
    if(COM.myrank==MASTER)
        printf("\n === done === \n");
    MPI_Finalize();
    return 0;
}